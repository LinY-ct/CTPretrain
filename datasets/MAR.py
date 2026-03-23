import os
import cv2
from PIL import Image
import random
import torch
import numpy as np
import glob
import h5py 
from torch.utils.data import Dataset
from wrappers.cttools import CTTools
from utilities.transformData import transformData
random.seed(42)

class MARDataset(Dataset):
    def __init__(self, sir_root, dataset_shape=256, spatial_dims=2, mode='train', use_num=None,
                  ):#de_type='all', dose=1, view=720
        assert mode in ['train', 'val', 'test'], f'Invalid mode: {mode}.'
        # print('de_type', de_type, 'dose', dose, 'view', view)
        # self.de_type = de_type
        # self.dose = dose
        # self.view = view
        self.mode = mode
        self.use_num = use_num
        self.hu_minv = -1024.0
        self.hu_maxv = 3071.0
        self.transformData = transformData()

        self.cttool = CTTools()
        if not isinstance(dataset_shape, (list, tuple)):
            dataset_shape = (dataset_shape,) * spatial_dims
        self.dataset_shape = dataset_shape

        self.filepath_list = self.get_path_list_from_dir(sir_root, use_num)
        print(f'finish loading Deeplession {mode} dataset, total images {len(self.filepath_list)}')


    def get_path_list_from_dir(self, src_dir, use_num=None):
        src_path_list = []
        if self.mode == 'train':
            src_path_list = os.listdir(os.path.join(src_dir, 'train'))
            src_path_list.remove('mask.h5')
            src_path_list.remove('metal_trace.h5')
            src_path_list = [os.path.join(src_dir, 'train', i) for i in src_path_list]
        elif self.mode == 'val':
            src_path_list = os.listdir(os.path.join(src_dir, 'test'))
            src_path_list.remove('mask.h5')
            src_path_list.remove('metal_trace.h5')
            src_path_list = [os.path.join(src_dir, 'test', i) for i in src_path_list]
        elif self.mode == 'test':
            src_path_list = os.listdir(os.path.join(src_dir, 'test'))
            src_path_list.remove('mask.h5')
            src_path_list.remove('metal_trace.h5')
            src_path_list = [os.path.join(src_dir, 'test', i) for i in src_path_list]
        if self.use_num is not None:
            return src_path_list[:self.use_num]
        return src_path_list

    def __getitem__(self, idx):
        data_path = self.filepath_list[idx]
        with h5py.File(data_path, 'r') as f:
            gt_CT_mu = f['gt_CT'][:]  # (416,416)
            ma_CT_mu = f['ma_CT'][:]  # (416,416)
            # gt_sinogram_water = f['gt_sinogram_water'][:] # (641, 640)
            # ma_sinogram = f['ma_sinogram'][:] # (641, 640) 

        W, H = gt_CT_mu.shape
        if H != self.dataset_shape[0] or W != self.dataset_shape[1]:
            gt_CT_mu = cv2.resize(gt_CT_mu, self.dataset_shape, cv2.INTER_CUBIC)
            ma_CT_mu = cv2.resize(ma_CT_mu, self.dataset_shape, cv2.INTER_CUBIC)
        
        MiuWater = 0.192
        gt_CT_hu = (gt_CT_mu - MiuWater) / MiuWater * 1000 
        ma_CT_hu = (ma_CT_mu - MiuWater) / MiuWater * 1000 
        gt_CT_hu[gt_CT_hu < -1024] = -1024
        gt_CT_hu[gt_CT_hu > 3071] = 3071
        ma_CT_hu[ma_CT_hu < -1024] = -1024
        ma_CT_hu[ma_CT_hu > 3071] = 3071

        ma_CT_mu = self.cttool.HU2mu(ma_CT_hu)
        gt_CT_mu = self.cttool.HU2mu(gt_CT_hu)
        ma_CT_mu = torch.from_numpy(ma_CT_mu).unsqueeze(0).unsqueeze(0).float()
        gt_CT_mu = torch.from_numpy(gt_CT_mu).unsqueeze(0).unsqueeze(0).float()
        if self.mode != 'test':
            gt_CT_mu, ma_CT_mu = self.transformData.pair_random_rotate_flip(gt_CT_mu, ma_CT_mu)

        gt_CT_mu = gt_CT_mu.squeeze(0)
        ma_CT_mu = ma_CT_mu.squeeze(0)
        return gt_CT_mu, ma_CT_mu, data_path.split('/')[-1]

    def __len__(self):
        return len(self.filepath_list)




