import torch
import pywt
import torch.nn as nn
import numpy as np

# 定义小波变换类（DWT 和 IDWT）
class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ll = torch.nn.functional.conv2d(x, self.w_ll.expand(C, -1, -1, -1), stride=2, groups=C)
        x_lh = torch.nn.functional.conv2d(x, self.w_lh.expand(C, -1, -1, -1), stride=2, groups=C)
        x_hl = torch.nn.functional.conv2d(x, self.w_hl.expand(C, -1, -1, -1), stride=2, groups=C)
        x_hh = torch.nn.functional.conv2d(x, self.w_hh.expand(C, -1, -1, -1), stride=2, groups=C)
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1).to(dtype=torch.float32)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1).to(dtype=torch.float32)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1).to(dtype=torch.float32)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1).to(dtype=torch.float32)
        self.filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = self.filters.repeat(C, 1, 1, 1)
        return torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)

# 示例数据
B, C, H, W = 1, 3, 256, 256  # 批量大小=1，通道数=3，图像大小=256x256
image = np.load("/mnt/nas/wsy/MayoData/npy/L333/full_1mm/L333_full_1mm_99.npy")
# image = np.expand_dims(image, axis=(0))
image = torch.tensor(image).to(dtype=torch.float32)
image = torch.stack([image, image, image], dim=1)
# image = torch.randn(B, C, H, W)  # 随机生成一个三通道图像

# 小波变换和逆小波变换
wave = 'haar'
dwt = DWT_2D(wave)
idwt = IDWT_2D(wave)

# 1. 小波变换
coeffs = dwt(image)  # 输出维度: (B, 4*C, H/2, W/2)

# 2. 分离 LL 和其他分量
LL = coeffs[:, :C, :, :]  # 低频部分
LH_HL_HH = coeffs[:, C:, :, :]  # 高频部分

# 3. 将高频部分置零，只保留 LL 重建
zero_high_freq = torch.zeros_like(LH_HL_HH)  # 高频置零
modified_coeffs = torch.cat([LL, zero_high_freq], dim=1)

# 将低频部分置零，只保留LH_HL_HH
zero_low_freq = torch.zeros_like(LL)
modified_coeffs_high = torch.cat([zero_low_freq, LH_HL_HH], dim=1)

# 4. 逆小波变换
low_reconstructed_image = idwt(modified_coeffs)  # 重建图像，维度: (B, C, H, W)
high_reconstructed_image = idwt(modified_coeffs_high)


print("输入图像维度:", image.shape)
print("小波分解后维度:", coeffs.shape)
print("重建high freq 图像维度:", high_reconstructed_image.shape)
print("重建low freq 图像维度:", low_reconstructed_image.shape)
import matplotlib.pyplot as plt
plt.imshow(image.numpy()[0, 0,:,:], cmap='gray')
plt.title('Original')
plt.show()
plt.imshow(low_reconstructed_image.numpy()[0, 0,:,:], cmap='gray')
plt.title('low Freq')
plt.show()
plt.imshow(high_reconstructed_image.numpy()[0, 0,:,:], cmap='gray')
plt.title('high Freq')
plt.show()