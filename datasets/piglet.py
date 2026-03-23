import sys
sys.path.append('..')

import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from wrappers.cttools import CTTools
random.seed(42)

class PigletDataset(Dataset):
    def __init__(self, sir_root, dataset_shape=256, spatial_dims=2, mode='train', use_num=None, returnType=False, test_file="test.txt"):
        assert mode in ['train', 'val', 'test'], f'Invalid mode: {mode}.'
        self.testfile = test_file
        self.returnType = returnType
        self.mode = mode
        self.use_num = use_num
        self.RescaleIntercept = -1024.0
        self.RescaleSlope = 1.0

        self.cttool = CTTools()
        if not isinstance(dataset_shape, (list, tuple)):
            dataset_shape = (dataset_shape,) * spatial_dims
        self.dataset_shape = dataset_shape
        self.type_all = ['5_dose','10_dose','25_dose', '50_dose']


        self.filepath_list = self.get_path_list_from_dir(sir_root, use_num)

        print(f'finish loading Piglet {mode} dataset, total images {len(self.filepath_list)}')


    def get_path_list_from_dir(self, src_dir, use_num=None):
        src_path_list = []
        if self.mode == 'train':
            with open(os.path.join(src_dir, 'train.txt'), "r") as file:
                for line in file:
                    src_path_list.append(line.strip())
        elif self.mode == 'val':
            with open(os.path.join(src_dir, 'test.txt'), "r") as file:
                for line in file:
                    src_path_list.append(line.strip())
        elif self.mode == 'test':
            with open(os.path.join(src_dir, self.testfile), "r") as file:
                for line in file:
                    src_path_list.append(line.strip())
        random.shuffle(src_path_list)
        if self.use_num is not None:
            return src_path_list[:self.use_num]
        return src_path_list

    def __getitem__(self, idx):
        target_path, input_path = self.filepath_list[idx].split(",")
        if os.path.basename(target_path) != os.path.basename(input_path):
            raise " target data not match input data"

        target_hu = np.load(target_path)  # dicom里面的数值
        target_hu = target_hu * self.RescaleSlope + self.RescaleIntercept
        target_hu[target_hu < -1024] = -1024
        target_hu[target_hu > 3071] = 3071
        W, H = target_hu.shape[-1], target_hu.shape[-2]
        if H != self.dataset_shape[0] or W != self.dataset_shape[1]:
            target_hu = cv2.resize(target_hu, self.dataset_shape, cv2.INTER_CUBIC)
        target_hu = torch.from_numpy(target_hu).unsqueeze(0).float()

        input_paths = []
        input_paths.append(input_path)
        type = input_path.split('/')[7]
        input_hus = []
        if self.returnType: # 读取其他噪声的数据
            for i in self.type_all:
                if i != type:
                    input_paths.append(input_path.replace(type, i))
        for i in input_paths:
            input_hu = np.load(i)
            input_hu = input_hu * self.RescaleSlope + self.RescaleIntercept
            input_hu[input_hu < -1024] = -1024
            input_hu[input_hu > 3071] = 3071
            if H != self.dataset_shape[0] or W != self.dataset_shape[1]:
                input_hu = cv2.resize(input_hu, self.dataset_shape, cv2.INTER_CUBIC)
            input_hu = torch.from_numpy(input_hu).unsqueeze(0).float()
            input_hus.append(input_hu)
        if self.returnType:
            return target_hu, input_hus[0], os.path.basename(target_path), input_hus[1],input_hus[2],input_hus[3]
        return target_hu, input_hus[0], os.path.basename(target_path)

    def __len__(self):
        return len(self.filepath_list)




