import random
import sys
sys.path.append('..')
import os
import torch
import torch.nn as nn
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from wrappers.cttools import CTTools
from trainers.basic_trainer import BasicTrainer
from datasets.aapm import AAPMMyoDataset
from datasets.deeplession import DeeplessionDataset
from utilities.transformData import transformData
from utilities.metrics import compute_measure
from utilities.schedulers import LinearWarmupCosineAnnealingLR
import torch.nn.functional as F
# from dinov2.models.vision_transformer import vit_large

def loading_pretrain_weight(model, path):
    # 获取当前模型的权重字典
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path)
    # 过滤掉不匹配的层
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict.keys())
    # 去掉patch_embed 和 output
    del pretrained_dict['patch_embed.proj.weight']
    del pretrained_dict['output.weight']
    # 更新当前模型的权重
    model_dict.update(pretrained_dict)
    # 将更新后的权重加载到模型中
    model.load_state_dict(model_dict)


class trainer_PIP(BasicTrainer):
    def __init__(self, opt, net):
        super().__init__()
        assert opt is not None and net is not None
        self.net = net
        self.opt = opt
        self.criterion = nn.L1Loss().cuda()
        self.itlog_intv = opt.log_interval
        self.cttool = CTTools()
        self.transformData = transformData()
        # self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()#torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        # state_dict = torch.load("/data01/dt_data/wsy/Codes/demo4/dinov2-main/pretrain/dinov2_vitl14_pretrain.pth")
        # self.dinov2_model.load_state_dict(state_dict, strict=True)
        self.data_range = {
            "aapm": 4095.0,
            'deeplession':12000.0
        }
        self.de_dict = {"ld": 0, "sv": 1, 'lv': 2, 'ld_sv': 3, 'ld_lv': 4}
        dataset_name = opt.dataset_name.lower()
        if dataset_name == 'aapm':
            self.train_dataset = AAPMMyoDataset(opt.dataset_path, mode='train',
                                                    dataset_shape=opt.dataset_shape,
                                                use_num=opt.num_train)
            self.val_dataset = AAPMMyoDataset(opt.dataset_path, mode='val',
                                                  dataset_shape=opt.dataset_shape,
                                                  use_num=opt.num_val)
        elif dataset_name == 'deeplession':
            self.train_dataset = DeeplessionDataset(opt.dataset_path,mode='train',
                                                    dataset_shape=opt.dataset_shape,
                                                    use_num=opt.num_train)
            self.val_dataset = DeeplessionDataset(opt.dataset_path, mode='val',
                                              dataset_shape=opt.dataset_shape,
                                              use_num=opt.num_val)
        else:
            raise NotImplementedError(f'Dataset {dataset_name} not implemented, try aapm.')

        # save path
        self.checkpoint_path = os.path.join(opt.checkpoint_root, opt.checkpoint_dir)
        if opt.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(opt.tensorboard_root, opt.tensorboard_dir))

    def fit(self):
        device = torch.device('cuda', 0)
        if self.opt.PretrainWeight != "":
            print("===="*20)
            print("加载自然图像预训练参数")
            print("====" * 20)
            loading_pretrain_weight(self.net,self.opt.PretrainWeight)
            # self.net.pretrain_load()

        # resume model
        if self.opt.resume:
            self.resume()
        else:
            try:
                self.weights_init(self.net)
            except Exception as err:
                print(f'init failed: {err}')

        self.net = self.net.to(device)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size,
                                       num_workers=self.opt.num_workers, shuffle=True,
                                       pin_memory=True, )
        self.val_loader = DataLoader(self.val_dataset, batch_size=1,
                                     num_workers=self.opt.num_workers, pin_memory=True)

        # 统计模型参数量
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad) / 1000
        print("Total Parameters (K):", total_params)

        # init and resume optimizer
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2),
                                           weight_decay=self.opt.weight_decay,)
        self.scheduler = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer,warmup_epochs=10, max_epochs=120, warmup_start_lr=self.opt.lr/10)

        if self.opt.resume_opt:
            self.resume_opt()


        # start training
        start_epoch = self.epoch
        self.iter = 0
        train_best_loss, val_best_loss = np.inf, np.inf
        for self.epoch in range(start_epoch, self.opt.epochs):
            print(f'start training epoch: {self.epoch}')
            train_current_loss = self.train()
            val_current_loss = self.val()
            self.scheduler.step()
            if train_best_loss > train_current_loss:
                self.save_model(net_name='best_train', loss=train_best_loss)
                self.save_opt(opt_name='best_train')
                train_best_loss = train_current_loss
            if val_best_loss > val_current_loss:
                self.save_model(net_name='best_val', loss=val_best_loss)
                self.save_opt(opt_name='best_val')
                val_best_loss = val_current_loss
            if ((self.epoch + 1) % self.opt.save_epochs == 0) or ((self.epoch + 1) == self.opt.epochs):
                self.save_model(net_name='current', loss=train_current_loss)
                self.save_opt(opt_name='current')

    def train(self):
        losses = []
        rmses, psnrs, ssims = [], [], []
        self.net = self.net.train()
        pbar = tqdm.tqdm(self.train_loader, ncols=160) if self.opt.use_tqdm else self.train_loader

        alpha = 0.002
        beta = 1
        if self.opt.high_reg_loss not in ['None', 'none', '']: # Default angle
            if 'bottle' in self.opt.network:
                reg_opt, reg_which, reg_which_ratio = self.opt.high_reg_loss, ['level_1', 'level_2', 'level_3', 'level4'], [1., 1., 1., 1.],
            elif 'Restormer' in self.opt.network:
                reg_opt, reg_which, reg_which_ratio = self.opt.high_reg_loss, ['level_1', 'level_2', 'level_3'], [1., 1., 1.],
            elif 'NAF' in self.opt.network:
                reg_opt, reg_which, reg_which_ratio = self.opt.high_reg_loss, ['level_1', 'level_2', 'level_3', 'level4'], [1., 1., 1., 1.],
            else:
                raise NotImplementedError

        for i, data in enumerate(pbar):
            gt_CT_mu, degenetaion_type, dose_range, view, filename = data
            gt_CT_mu = gt_CT_mu.to('cuda')
            gt_CT_mu, ma_CT_mu, gt_sinogram, ma_sinogram, mask_sinogram = \
                self.net.make_traindata2(gt_CT_mu, gt_CT_mu, degenetaion_type[0], dose_range[0].item(), view[0].item())
            gt_CT_hu, ma_CT_hu = self.cttool.mu2HU(gt_CT_mu), self.cttool.mu2HU(ma_CT_mu)

            # import matplotlib.pyplot as plt
            # plt.imshow(gt_CT_hu.cpu().numpy()[0,0,:,:], cmap='gray')
            # plt.show()
            # plt.imshow(ma_CT_hu.cpu().numpy()[0, 0, :, :], cmap='gray')
            # plt.show()

            gt_CT_hu = torch.cat([gt_CT_hu,gt_CT_hu,gt_CT_hu], dim=1)
            ma_CT_hu = torch.cat([ma_CT_hu, ma_CT_hu, ma_CT_hu], dim=1)

            ma_CT_norm = self.transformData.normalize(ma_CT_hu)
            gt_CT_norm = self.transformData.normalize(gt_CT_hu)

            de_id = torch.tensor(self.de_dict[degenetaion_type[0]])
            #print('degradation_class:',de_id)
            restored ,basic_prompt1, basic_prompt2,basic_prompt3, predictions = self.net(ma_CT_norm, degradation_class = de_id)

            # dino_HQ_Features = self.dinov2_model.get_intermediate_layers(F.interpolate(gt_CT_norm, size=(224, 224), mode='bilinear', align_corners=False), n=[6, 12, 18], reshape=False)
            if 'pip' in self.opt.network.lower():
                # stable training avoid large loss
                pixel_loss = self.scale_pixel_loss(restored, gt_CT_norm, mode='sml1')
                # pixel_loss = 0.7*self.scale_pixel_loss(restored, gt_CT_norm, mode='sml1') + 0.3*(
                #     self.scale_pixel_loss(dino_HQ_Features[0], basic_prompt1, mode='sml1')+
                #     self.scale_pixel_loss(dino_HQ_Features[1], basic_prompt2, mode='sml1')+
                #     self.scale_pixel_loss(dino_HQ_Features[2], basic_prompt2, mode='sml1')
                # )

                # reg loss
                if self.opt.high_reg_loss not in ['None', 'none', '']:
                    param_reg_loss = self.net.param_regulaztion_loss(reg_opt=reg_opt, reg_which=reg_which,
                                                                            reg_which_ratio=reg_which_ratio)
                else:
                    param_reg_loss = torch.tensor(0)

                pred_loss1 = pred_loss2 = pred_loss3 = torch.tensor(0)
                pred_loss = pred_loss1 + pred_loss2 + pred_loss3
                loss = pixel_loss + param_reg_loss * alpha + pred_loss * beta
            else:
                raise NotImplementedError(f"not support network:{self.opt.network} for pip training")


            # loss = self.criterion(restored, gt_CT_norm)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            # restored_dnorm = self.transformData.denormalize(restored, self.opt.dataset_name.lower())
            rmse, psnr, ssim = compute_measure(restored, gt_CT_norm,data_range=1)
            rmses.append(rmse)
            psnrs.append(psnr)
            ssims.append(ssim)
            if self.opt.use_tqdm:
                pbar.set_postfix(
                    {'epoch': '%d' % (self.epoch), 'loss': '%.6f' % (loss.item()), 'psnr': '%.2f' % (psnr),
                     'ssim': '%.2f' % (ssim)})
                pbar.update(1)
            # log acc by iteration
            if self.iter != 0 and self.iter % self.itlog_intv == 0:
                log_info = {
                    'loss': np.mean(losses[-self.itlog_intv:]),
                    'rmse': np.mean(rmses[-self.itlog_intv:]),
                    'ssim': np.mean(ssims[-self.itlog_intv:]),
                    'psnr': np.mean(psnrs[-self.itlog_intv:]),}
                if self.opt.use_tensorboard:
                    self.tensorboard_scalar(self.writer, 'train/loss', self.iter, **log_info)
            self.iter += 1
        # epoch info
        epoch_log = {
            'loss': np.mean(losses),
            'rmse': np.mean(rmses),
            'ssim': np.mean(ssims),
            'psnr': np.mean(psnrs),}
        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        print(f'Epoch {self.epoch} learning rate: {current_lr}')
        print(f'Epoch {self.epoch} train loss: {epoch_log["loss"]}')
        print(f'Epoch {self.epoch} train rmse: {epoch_log["rmse"]}')
        print(f'Epoch {self.epoch} train psnr: {epoch_log["psnr"]}')
        print(f'Epoch {self.epoch} train ssim: {epoch_log["ssim"]}')
        if self.opt.use_tensorboard:
            self.tensorboard_scalar(self.writer, 'train/epoch', self.epoch, **epoch_log)
            self.tensorboard_scalar(self.writer, 'settings', self.epoch,
                                    **{'current_lr': current_lr, 'batch_size': self.opt.batch_size})
        return epoch_log["loss"]

    def val(self):
        losses = []
        rmses, psnrs, ssims = [], [], []
        self.net = self.net.eval()
        pbar = tqdm.tqdm(self.val_loader, ncols=160) if self.opt.use_tqdm else self.val_loader
        with torch.no_grad():
            for i, data in enumerate(pbar):
                gt_CT_mu,degenetaion_type, dose_range, view ,filename = data
                gt_CT_mu = gt_CT_mu.to('cuda')
                gt_CT_mu, ma_CT_mu, gt_sinogram, ma_sinogram, mask_sinogram = \
                    self.net.make_traindata2(gt_CT_mu, gt_CT_mu, degenetaion_type[0], dose_range[0].item(), view[0].item())
                gt_CT_hu, ma_CT_hu = self.cttool.mu2HU(gt_CT_mu), self.cttool.mu2HU(ma_CT_mu)
                gt_CT_hu = torch.cat([gt_CT_hu, gt_CT_hu, gt_CT_hu], dim=1)
                ma_CT_hu = torch.cat([ma_CT_hu, ma_CT_hu, ma_CT_hu], dim=1)
                ma_CT_norm = self.transformData.normalize(ma_CT_hu)
                gt_CT_norm = self.transformData.normalize(gt_CT_hu)

                de_id = torch.tensor(self.de_dict[degenetaion_type[0]])
                restored, basic_prompt1, basic_prompt2,basic_prompt3, predictions  = self.net(ma_CT_norm, degradation_class = de_id)
                loss = self.criterion(restored, gt_CT_norm)

                losses.append(loss.item())
                # restored_dnorm = self.transformData.denormalize(restored, self.opt.dataset_name.lower())
                rmse, psnr, ssim = compute_measure(restored, gt_CT_norm,data_range=1)
                rmses.append(rmse)
                psnrs.append(psnr)
                ssims.append(ssim)
                if self.opt.use_tqdm:
                    pbar.set_postfix(
                        {'epoch': '%d' % (self.epoch), 'loss': '%.6f' % (loss.item()), 'psnr': '%.2f' % (psnr),
                         'ssim': '%.2f' % (ssim)})
                    pbar.update(1)
                # log acc by iteration
                if self.iter != 0 and self.iter % self.itlog_intv == 0:
                    log_info = {
                        'loss': np.mean(losses[-self.itlog_intv:]),
                        'rmse': np.mean(rmses[-self.itlog_intv:]),
                        'ssim': np.mean(ssims[-self.itlog_intv:]),
                        'psnr': np.mean(psnrs[-self.itlog_intv:]),}
                    if self.opt.use_tensorboard:
                        self.tensorboard_scalar(self.writer, 'val/loss', self.iter, **log_info)

                self.iter += 1
        # epoch info
        epoch_log = {
            'loss': np.mean(losses),
            'rmse': np.mean(rmses),
            'ssim': np.mean(ssims),
            'psnr': np.mean(psnrs),}
        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        print(f'Epoch {self.epoch} learning rate: {current_lr}')
        print(f'Epoch {self.epoch} val loss: {epoch_log["loss"]}')
        print(f'Epoch {self.epoch} val rmse: {epoch_log["rmse"]}')
        print(f'Epoch {self.epoch} val psnr: {epoch_log["psnr"]}')
        print(f'Epoch {self.epoch} val ssim: {epoch_log["ssim"]}')
        if self.opt.use_tensorboard:
            self.tensorboard_scalar(self.writer, 'val/epoch', self.epoch, **epoch_log)
            self.tensorboard_scalar(self.writer, 'settings', self.epoch,
                                    **{'current_lr': current_lr, 'batch_size': self.opt.batch_size})
        return epoch_log["loss"]
