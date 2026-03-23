import os
import tqdm
import random
import numpy as np
import pandas as pd
import scipy.io
import torch
from collections import defaultdict
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from utilities.transformData import transformData
torch.multiprocessing.set_sharing_strategy('file_system')
import sys

sys.path.append('..')
from trainers.basic_trainer import BasicTrainer
from wrappers.cttools import CTTools
from utilities.metrics import compute_measure
from datasets.aapm import AAPMMyoDataset
from datasets.deeplession import DeeplessionDataset

class tester_PIP(BasicTrainer):
    def __init__(self, opt, net):
        super().__init__()
        self.opt = opt
        self.net = net
        self.cttool = CTTools()
        self.transformData = transformData()
        self.save_fig = self.opt.tester_save_image
        self.save_dir = os.path.join(self.opt.tester_save_path, self.opt.tester_save_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f'Save figures to {self.save_dir}? : ', self.save_fig)
        self.saved_slice = 0
        self.seed_torch(seed=1)
        self.data_range = {
            "aapm": 4095.0,
            'deeplession': 6000.0
        }
        self.de_dict = {"ld": 0, "sv": 1, 'lv': 2, 'ld_sv': 3, 'ld_lv': 4}

    def seed_torch(self, seed=1):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def prepare_dataset(self, ):
        opt = self.opt
        dataset_name = opt.dataset_name.lower()
        if dataset_name == 'aapm':
            self.test_dataset = AAPMMyoDataset(opt.dataset_path, mode='test',
                                             dataset_shape=opt.dataset_shape,
                                               de_type=opt.de_type,
                                               dose=opt.dose,
                                               view=opt.view
                                                )
        elif dataset_name == 'deeplession':
            self.test_dataset = DeeplessionDataset(opt.dataset_path, mode='test',
                                             dataset_shape=opt.dataset_shape,
                                                   de_type=opt.de_type,
                                               dose=opt.dose,
                                               view=opt.view)
        else:
            raise NotImplementedError(f'Dataset {dataset_name} not implemented, try aapm.')
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, num_workers=8, )
        if self.opt.tester_save_matnum is None:
            self.opt.tester_save_matnum = len(self.test_dataset)

    def prepare(self, *args):
        device = torch.device('cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def run(self, ):
        self.prepare_dataset()

        cl = ['filename']

        for k in ['psnr', 'ssim', 'rmse']:
            for i in ['gt-inputput-ct-', 'gt-finaloutput-ct-']:
                for j in ['norm_']:  # 'norm_',
                        cl.append(i + j + k)

        self.tables = pd.DataFrame(columns=cl)
        self.tables_stat = pd.DataFrame(columns=cl)

        if self.opt.network != 'fbp':
            # net_checkpoint = torch.load(self.opt.net_checkpath, map_location='cpu')
            # net_checkpoint = net_checkpoint['net_param'] if 'net_param' in net_checkpoint.keys() else net_checkpoint
            # self.net.model.load_state_dict(net_checkpoint, strict=False)
            self.load_model()

        self.net = self.net.cuda()
        self.net = self.net.train()
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad) / 1000000
        print("Total Parameters (M):", total_params)
        self.net = self.net.eval()
        pbar = tqdm.tqdm(self.test_loader, ncols=60)

        record = {}

        with torch.no_grad():
            for i, data in enumerate(pbar):
                gt_CT_mu, degenetaion_type, dose_range, view, filename = data
                gt_CT_mu = gt_CT_mu.to('cuda')
                gt_CT_mu, ma_CT_mu, gt_sinogram, ma_sinogram, mask_sinogram, degenetaion_type, value_str = \
                    self.net.make_traindata2(gt_CT_mu, gt_CT_mu, degenetaion_type[0], dose_range[0].item(), view[0].item(), return_choice=True)
                gt_CT_hu, ma_CT_hu = self.cttool.mu2HU(gt_CT_mu), self.cttool.mu2HU(ma_CT_mu)

                gt_CT_hu = torch.cat([gt_CT_hu, gt_CT_hu, gt_CT_hu], dim=1)
                ma_CT_hu = torch.cat([ma_CT_hu, ma_CT_hu, ma_CT_hu], dim=1)

                ma_CT_norm = self.transformData.normalize(ma_CT_hu)
                gt_CT_norm = self.transformData.normalize(gt_CT_hu)
                # import matplotlib.pyplot as plt
                # plt.imshow(gt_CT_hu.cpu().numpy()[0, 0, :, :], cmap='gray')
                # plt.title('GT')
                # plt.show()
                # plt.imshow(ma_CT_hu.cpu().numpy()[0, 0, :, :], cmap='gray')
                # plt.title('ma_CT_hu')
                # plt.show()
                # plt.imshow(ma_CT_norm.cpu().numpy()[0, 0, :, :], cmap='gray')
                # plt.title('ma_CT_norm')
                # plt.show()
                de_id = torch.tensor(self.de_dict[degenetaion_type])
                restored, predictions  = self.net(ma_CT_norm, degradation_class = de_id)
                restored_hu = self.transformData.denormalize(restored)

                # import matplotlib.pyplot as plt
                # plt.imshow(gt_CT_hu.cpu().numpy()[0, 0, :, :], cmap='gray')
                # plt.title('GT')
                # plt.show()
                # plt.imshow(ma_CT_hu.cpu().numpy()[0, 0, :, :], cmap='gray')
                # plt.title('ma_CT_hu')
                # plt.show()
                # plt.imshow(ma_CT_norm.cpu().numpy()[0, 0, :, :], cmap='gray')
                # plt.title('ma_CT_norm')
                # plt.show()
                # plt.imshow(restored.cpu().numpy()[0, 0, :, :], cmap='gray')
                # plt.title('restored')
                # plt.show()
                record['filename'] = filename[0]
                record['de_type'] = degenetaion_type
                record['de_value'] = value_str
                rmse, psnr, ssim = compute_measure(restored, gt_CT_norm, data_range=1)
                record['gt-finaloutput-ct-norm_rmse'] = rmse
                record['gt-finaloutput-ct-norm_psnr'] = psnr
                record['gt-finaloutput-ct-norm_ssim'] = ssim

                rmse, psnr, ssim = compute_measure(ma_CT_norm, gt_CT_norm, data_range=1)
                record['gt-inputput-ct-norm_rmse'] = rmse
                record['gt-inputput-ct-norm_psnr'] = psnr
                record['gt-inputput-ct-norm_ssim'] = ssim
                self.tables = self.tables.append(record, ignore_index=True)

                gt_CT_hu = gt_CT_hu.permute(0,2,3,1)
                ma_CT_hu = ma_CT_hu.permute(0, 2, 3, 1)
                restored_hu = restored_hu.permute(0, 2, 3, 1)
                mdict = {'gt_ct': gt_CT_hu[:,:,:,0:1], 'de_ct': ma_CT_hu[:,:,:,0:1], 'filename': filename, 'output': restored_hu[:,:,:,0:1]}
                if self.save_fig and self.saved_slice < self.opt.tester_save_matnum:
                    self.save_mat(**mdict)
                self.saved_slice += 1
        self.save_csv()

    def save_mat(self, **mdict):
        data_save = {}
        for key, value in mdict.items():
            if key not in ['filename', 'de_value', 'de_type']:
                data_save[key] = value.cpu().squeeze_().data.numpy()
            else:
                data_save[key] = value
        os.makedirs(self.save_dir, exist_ok=True)
        data_name = "".join(mdict['filename']).split('.')[0]
        scipy.io.savemat('{}/{}.mat'.format(self.save_dir, data_name), mdict=data_save)

    def save_csv(self, ):
        csv_path = os.path.join(self.save_dir, self.opt.network + '_all.csv')
        self.tables.to_csv(csv_path, index=False)
        statistics_df = self.tables.describe()
        csv_stat_path = os.path.join(self.save_dir, self.opt.network  + '_stat.csv')
        statistics_df.to_csv(csv_stat_path, index=True)
