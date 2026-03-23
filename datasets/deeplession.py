import os
import cv2
from PIL import Image
import random
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
from wrappers.cttools import CTTools
from utilities.transformData import transformData
random.seed(42)

class DeeplessionDataset(Dataset):
    def __init__(self, sir_root, dataset_shape=256, spatial_dims=2, mode='train', use_num=None,
                 de_type='all', dose=1, view=720 ):
        assert mode in ['train', 'val', 'test'], f'Invalid mode: {mode}.'
        print('de_type', de_type, 'dose', dose, 'view', view)
        self.de_type = de_type
        self.dose = dose
        self.view = view
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
            with open(os.path.join(src_dir, 'train.txt'), "r") as file:
                for line in file:
                    src_path_list.append(line.strip())
        elif self.mode == 'val':
            with open(os.path.join(src_dir, 'val.txt'), "r") as file:
                for line in file:
                    src_path_list.append(line.strip())
        elif self.mode == 'test':
            with open(os.path.join(src_dir, 'test.txt'), "r") as file:
                for line in file:
                    src_path_list.append(line.strip())
        if self.use_num is not None:
            return src_path_list[:self.use_num]
        return src_path_list

    def __getitem__(self, idx):
        line = self.filepath_list[idx].split(",")
        data_path = line[0]
        degenetaion_type = line[1]
        dose_range = line[2]
        view = line[3]

        img = Image.open(data_path)

        W, H = img.size
        if H != self.dataset_shape[0] or W != self.dataset_shape[1]:
            img = img.resize(self.dataset_shape, Image.BICUBIC)

        img = np.asarray(img)
        image_hu = (img - 32768)
        image_hu[image_hu < -1024] = -1024
        image_hu[image_hu > 3071] = 3071

        image_mu = self.cttool.HU2mu(image_hu)
        image_mu = torch.from_numpy(image_mu).unsqueeze(0).unsqueeze(0).float()
        if self.mode != 'test':
            image_mu = self.transformData.random_rotate_flip(image_mu)
        
        if self.de_type != 'all':
            degenetaion_type = self.de_type
            dose_range = self.dose
            view = self.view
        image_mu = image_mu.squeeze(0)
        return image_mu , degenetaion_type, int(dose_range), int(view), data_path.split('/')[-1]

    def __len__(self):
        return len(self.filepath_list)




