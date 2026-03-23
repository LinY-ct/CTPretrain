import random
import torch
class transformData:
    def __init__(self, data_range='aapm'):
        self.r = data_range
        self.data_range = {
            "CT":[-1024.0, 3071.0],
            }
    def truncate(self,img, d_min, d_max):
        img[img>d_max]=d_max
        img[img<d_min]=d_min
        return img

    def normalize(self, img, modality='CT'):
        d_min, d_max = self.data_range[modality]
        img = self.truncate(img, d_min, d_max)
        img = (img - d_min)/(d_max - d_min)
        return img

    def denormalize(self, img, modality='CT'):
        d_min, d_max = self.data_range[modality]
        img = img*(d_max - d_min) + d_min
        img = self.truncate(img, d_min, d_max)
        return img

    def random_crop(self, tensor, patch_size):
        """
        从给定的图像张量中随机裁剪大小为patch_size的patch。

        参数:
        tensor: 形状为[B, C, H, W]的图像张量。
        patch_size: 裁剪patch的大小，格式为(H, W)。

        返回:
        裁剪后的patch张量。
        """
        B, C, H, W = tensor.shape
        patch_h, patch_w = patch_size

        # 确保裁剪尺寸不大于原图像尺寸
        if patch_h > H or patch_w > W:
            raise ValueError("裁剪尺寸应小于原始图像尺寸")

        # 随机选择裁剪的起始点
        top = random.randint(0, H - patch_h)
        left = random.randint(0, W - patch_w)

        # 裁剪patch
        patches = tensor[:, :, top:top + patch_h, left:left + patch_w]
        return patches

    def random_rotate_flip(self, tensor):
        """
        对形状为[B, C, H, W]的图像张量执行随机旋转或翻转。

        参数:
        tensor: 形状为[B, C, H, W]的图像张量。

        返回:
        经过随机旋转或翻转的图像张量。
        """
        B, C, H, W = tensor.shape
        processed = torch.empty_like(tensor)

        for i in range(B):
            img = tensor[i]
            operation = torch.randint(0, 6, (1,)).item()

            if operation == 1:
                # 水平翻转
                img = torch.flip(img, [2])
            elif operation == 2:
                # 垂直翻转
                img = torch.flip(img, [1])
            elif operation == 3:
                # 旋转90度
                img = img.transpose(1, 2).flip(2)
            elif operation == 4:
                # 旋转180度
                img = img.flip(1).flip(2)
            elif operation == 5:
                # 旋转270度
                img = img.transpose(1, 2).flip(1)

            # 不做改变的情况下，operation == 0
            processed[i] = img

        return processed
    def pair_random_rotate_flip(self, tensor1, tensor2):
        """
        对形状为[B, C, H, W]的图像张量执行随机旋转或翻转。

        参数:
        tensor: 形状为[B, C, H, W]的图像张量。

        返回:
        经过随机旋转或翻转的图像张量。
        """
        B, C, H, W = tensor1.shape
        processed1 = torch.empty_like(tensor1)
        processed2 = torch.empty_like(tensor2)

        for i in range(B):
            img1 = tensor1[i]
            img2 = tensor2[i]
            operation = torch.randint(0, 6, (1,)).item()

            if operation == 1:
                # 水平翻转
                img1 = torch.flip(img1, [2])
                img1 = torch.flip(img2, [2])
            elif operation == 2:
                # 垂直翻转
                img1 = torch.flip(img1, [1])
                img2 = torch.flip(img2, [1])
            elif operation == 3:
                # 旋转90度
                img1 = img1.transpose(1, 2).flip(2)
                img2 = img2.transpose(1, 2).flip(2)
            elif operation == 4:
                # 旋转180度
                img1 = img1.flip(1).flip(2)
                img2 = img2.flip(1).flip(2)
            elif operation == 5:
                # 旋转270度
                img1 = img1.transpose(1, 2).flip(1)
                img2 = img2.transpose(1, 2).flip(1)

            # 不做改变的情况下，operation == 0
            processed1[i] = img1
            processed2[i] = img2

        return processed1, processed2