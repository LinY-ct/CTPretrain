import os
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import utilities
import sys

sys.path.append('..')
from wrappers.cttools import CTTools
from utilities.metrics import compute_measure
from collections import deque

class LossScaler:
    def __init__(self, max_len=50, scale_factor=2):
        """
        intro:
            scaler large loss to stable training
        """
        self.history = deque(maxlen=max_len)
        self.scale_factor = scale_factor

    def __call__(self, loss):
        self.history.append(loss.detach().item())
        min_loss = min(self.history)

        # scale
        if loss.item() > min_loss * self.scale_factor:
            scaled_loss = loss * (min_loss * (self.scale_factor/2) / loss.item())
            return scaled_loss
        else:
            return loss

class BasicTrainer:
    def __init__(self):
        self.iter = 0
        self.epoch = 0
        self.pre_epoch = 0
        self.cttool = CTTools()
        self.loss_scaler = LossScaler(max_len=100, scale_factor=5)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def get_optimizer(self, net):
        opt = self.opt
        optimizer_name = opt.optimizer
        assert isinstance(optimizer_name, str)
        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adam':
            return torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, self.opt.beta2))
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        elif optimizer_name == 'lbfgs':
            return torch.optim.LBFGS(net.parameters(), lr=opt.lr, tolerance_grad=-1, tolerance_change=-1)
        else:
            raise NotImplementedError(
                f'Currently only support optimizers among Adam/AdamW/SGD/LBFGS, got {optimizer_name}.')

    def get_scheduler(self, optimizer):
        opt = self.opt
        scheduler_name = opt.scheduler
        assert isinstance(scheduler_name, str)
        scheduler_name = scheduler_name.lower()
        if scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.step_gamma)
        elif scheduler_name == 'mstep':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.step_gamma)
        elif scheduler_name == 'exp':
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.step_gamma)
        elif scheduler_name == 'poly':
            return optim.lr_scheduler.PolynomialLR(optimizer, total_iters=opt.poly_iters, power=opt.poly_power)
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)
        else:
            raise NotImplementedError(
                f'Currently only support schedulers among Step/MultiStep/Exp/Poly/Cosine, got {scheduler_name}.')

    def reduce_value(self, value, average=True):
        world_size = dist.get_world_size()
        if world_size < 2:  # single GPU
            return value
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if not value.is_cuda:
            value = value.cuda(self.opt.local_rank)
        with torch.no_grad():
            dist.all_reduce(value)  # get reduce value
            if average:
                value = value.float()
                value /= world_size
        return value.cpu()

    @staticmethod
    def save_checkpoint(param, path, name: str):
        # simply save the checkpoint by epoch
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, name + '.pth')
        torch.save(param, checkpoint_path)

    def save_model(self, net=None, net_name='', loss=np.inf):
        net_param = net.state_dict() if net is not None else self.net.state_dict()  # default multicard
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        net_check = {'net_param': net_param, 'epoch': self.epoch, 'loss': loss, 'pre_epoch': self.pre_epoch}
        print('save {} model, epoch: {}, pre_epoch: {}'.format(net_name, self.epoch, self.pre_epoch))
        self.save_checkpoint(net_check, checkpoint_path, self.opt.checkpoint_dir + '-net-' + net_name)

    def save_opt(self, optimizer=None, scheduler=None, opt_name=''):
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        optimizer_param = optimizer.state_dict() if optimizer is not None else self.optimizer.state_dict()
        scheduler_param = scheduler.state_dict() if scheduler is not None else self.scheduler.state_dict()
        opt_check = {
            'optimizer': optimizer_param,
            'scheduler': scheduler_param,
            'epoch': self.epoch,
            'pre_epoch': self.pre_epoch
        }
        self.save_checkpoint(opt_check, checkpoint_path, self.opt.checkpoint_dir + '-opt-' + opt_name)

    def load_model(self, net=None, net_checkpath=None, output=False):
        net_checkpath = self.opt.net_checkpath if net_checkpath is None else net_checkpath
        net_checkpoint = torch.load(net_checkpath, map_location='cpu')
        epoch = net_checkpoint['epoch'] if 'epoch' in net_checkpoint.keys() else 0
        net_checkpoint = net_checkpoint['net_param'] if 'net_param' in net_checkpoint.keys() else net_checkpoint
        if net is None:
            self.net.load_state_dict(net_checkpoint, strict=True)
        else:
            net.load_state_dict(net_checkpoint, strict=False)
        print('finish loading network, epoch: ', epoch)
        if output:
            return net

    def load_pretrain_model(self, net=None, net_checkpath=None, output=None):
        net_checkpath = self.opt.net_checkpath if net_checkpath is None else net_checkpath
        net_checkpoint = torch.load(net_checkpath, map_location='cpu')
        net_checkpoint = net_checkpoint['net_param'] if 'net_param' in net_checkpoint.keys() else net_checkpoint
        if net is None:
            model_dict = self.net.state_dict()
            # 从预训练参数中去掉不匹配的参数
            pretrained_dict = {k: v for k, v in net_checkpoint.items() if k in model_dict}

            # 更新当前模型的参数字典
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict, strict=True)
        else:
            model_dict = net.state_dict()
            # 从预训练参数中去掉不匹配的参数
            pretrained_dict = {k: v for k, v in net_checkpoint.items() if k in model_dict}

            # 更新当前模型的参数字典
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict, strict=True)
        print('finish loading network')
        if output:
            return net

    def load_opt(self):
        opt_checkpath = self.opt.opt_checkpath
        opt_checkpoint = torch.load(opt_checkpath, map_location='cpu')
        self.optimizer.load_state_dict(opt_checkpoint['optimizer'])
        self.scheduler.load_state_dict(opt_checkpoint['scheduler'])
        self.epoch = opt_checkpoint['epoch']
        if 'pre_epoch' in opt_checkpoint.keys():
            self.pre_epoch = opt_checkpoint['pre_epoch']
        print('finish loading opt')

    def resume(self, net=None):
        if self.opt.net_checkpath is not None:
            self.load_model()
        else:
            raise ValueError('opt.net_checkpath not provided.')

    def resume_opt(self, ):
        if self.opt.resume_opt and self.opt.opt_checkpath is not None:
            self.load_opt()
            print('finish loading optimizer')
        else:
            print('opt.opt_checkpath not provided')

    def get_pixel_criterion(self, mode=None, reduction='mean'):
        mode = self.opt.loss_type if mode is None else mode
        assert isinstance(mode, str)
        mode = mode.lower()
        if mode == 'l1':
            criterion = torch.nn.L1Loss(reduction=reduction)
        elif mode == 'sml1':
            criterion = torch.nn.SmoothL1Loss(reduction=reduction)
        elif mode == 'l2':
            criterion = torch.nn.MSELoss(reduction=reduction)
        elif mode in ['crossentropy', 'ce', 'cross_entropy']:
            criterion = torch.nn.CrossEntropyLoss()
        elif mode in ['cos', 'consine']:
            def cosine_similarity_loss(x, y):
                sim = torch.cosine_similarity(x, y, dim=0)
                if reduction == 'mean':
                    return 1. - sim.mean()
                else:
                    return 1. - sim.sum()

            criterion = cosine_similarity_loss
        else:
            raise NotImplementedError('pixel_loss error: mode not in [l1, sml1, l2, ce].')
        return criterion

    # ---- basic logging function ----
    @staticmethod
    def tensorboard_scalar(writer, rel_path, step, **kwargs):
        for key in kwargs.keys():
            path = os.path.join(rel_path, key)
            writer.add_scalar(path, kwargs[key], global_step=step)

    @staticmethod
    def tensorboard_image(writer, rel_path, **kwargs):
        for key in kwargs.keys():
            path = os.path.join(rel_path, key)
            writer.add_image(tag=path, img_tensor=kwargs[key], global_step=0, dataformats='CHW', )

    @staticmethod
    def wandb_init(opt, key=None):
        if key is None:
            print('WANDB key not provided, attempting anonymous login...')
        else:
            wandb.login(key=key)
        wandb_root = opt.tensorboard_root if opt.wandb_root == '' else opt.wandb_root
        wandb_dir = opt.tensorboard_dir if opt.wandb_dir == '' else opt.wandb_dir
        wandb_path = os.path.join(wandb_root, wandb_dir)
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)
        wandb.init(project=opt.wandb_project, name=str(wandb_dir), dir=wandb_path, resume='allow', reinit=True, )

    @staticmethod
    def to_wandb_img(**kwargs):
        # turn torch makegrid to wandb image
        for key, value in kwargs.items():
            kwargs[key] = wandb.Image(kwargs[key].float().cpu())
        return kwargs

    @staticmethod
    def wandb_logger(r_path=None, step_name=None, step=None, **kwargs):
        log_info = {}
        if step is not None:
            log_info.update({str(step_name): step})
        for key, value in kwargs.items():
            if r_path is not None:
                key_name = str(os.path.join(r_path, key))
            else:
                key_name = key
            log_info[key_name] = kwargs[key]
        wandb.log(log_info)

    @staticmethod
    def wandb_scalar(r_path, step=None, **kwargs):
        for key in kwargs.keys():
            if step is not None:
                wandb.log({'{}'.format(os.path.join(r_path, key)): kwargs[key]}, step=step)
            else:
                wandb.log({'{}'.format(os.path.join(r_path, key)): kwargs[key]})

    @staticmethod
    def wandb_image(step=None, **kwargs):
        for key in kwargs.keys():
            kwargs[key] = wandb.Image(kwargs[key].float().cpu())
        if step is not None:
            wandb.log(kwargs, step=step)
        else:
            wandb.log(kwargs)

    # basic Sparse_CT accuracy by window
    def get_metrics_by_window(self, mu_input, mu_target, to_HU=True):
        # calculate mu ct accuracy by CT window
        if to_HU:
            hu_input = self.cttool.window_transform(self.cttool.mu2HU(mu_input))
            hu_target = self.cttool.window_transform(self.cttool.mu2HU(mu_target))
        else:
            hu_input, hu_target = mu_input, mu_target
        # data_range of normalized HU img
        data_range = 1
        rmse, psnr, ssim = compute_measure(hu_input, hu_target, data_range)
        return rmse, psnr, ssim

    def get_sinogram_metrics_by_norm(self, mu_input, mu_traget):
        minv, maxv = -0.1, 7.0
        mu_input = (mu_input - minv) / (maxv - minv)
        mu_input[mu_input < 0] = 0
        mu_input[mu_input > 1] = 1
        mu_traget = (mu_traget - minv) / (maxv - minv)
        mu_traget[mu_traget < 0] = 0
        mu_traget[mu_traget > 1] = 1
        rmse, psnr, ssim = compute_measure(mu_input, mu_traget, data_range=1)
        return rmse, psnr, ssim

    def normalization(self, data):
        minv, maxv = torch.min(data), torch.max(data)
        return (data - torch.min(data)) / (maxv - minv)

    def hu2Normalization(self, data):
        minv, maxv = -1024.0, 3072.0
        data_norm = (data - minv) / (maxv - minv)
        data_norm[data_norm < 0] = 0
        data_norm[data_norm > 1] = 1
        return data_norm

    def get_metrics_by_norm(self, mu_input, mu_target, to_HU=True, to_norm=True):

        if to_HU:
            mu_input = self.cttool.mu2HU(mu_input)
            mu_target = self.cttool.mu2HU(mu_target)

        if to_norm:
            mu_input = mu_input.double()
            mu_target = mu_target.double()
            mu_input, mu_target = self.hu2Normalization(mu_input), self.hu2Normalization(mu_target)

        rmse, psnr, ssim = compute_measure(mu_input, mu_target, data_range=1)
        return rmse, psnr, ssim

    def loading_pretrain_weight(self, model,path):
        # 获取当前模型的权重字典
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        # 过滤掉不匹配的层
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print(pretrained_dict.keys())
        # 更新当前模型的权重
        model_dict.update(pretrained_dict)
        # 将更新后的权重加载到模型中
        model.load_state_dict(model_dict)

    # ---- training fucntion ----
    def fit(self):
        raise NotImplementedError('function fit() not implemented.')

    def train(self):
        raise NotImplementedError('function train() not implemented.')

    def val(self):
        raise NotImplementedError('function val() not implemented.')

    def scale_pixel_loss(self, input, target, mode = 'l1'):
        assert mode in ['l1', 'sml1', 'l2']
        if mode == 'l1':
            L1loss = torch.nn.L1Loss(reduction = 'mean')
            loss = L1loss(input, target)
        elif mode == 'sml1':
            smL1loss = torch.nn.SmoothL1Loss(reduction = 'mean')
            loss = smL1loss(input, target)
        elif mode == 'l2':
            mse_loss = torch.nn.MSELoss(reduction = 'mean')
            loss = mse_loss(input, target)
        else:
            raise ValueError('pixel_loss error: mode not in [l1,sml1,l2]')
        return self.loss_scaler(loss)