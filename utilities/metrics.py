import torch
import numpy as np
from math import exp
import torch.nn.functional as F

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def compute_measure(pred, y, data_range, spatial_dims=2):
    B,C,H,W = pred.size()
    pred_psnr = compute_PSNR(pred, y, data_range)
    pred_ssim = 0
    for i in range(C):
        pred_ssim += compute_SSIM(pred[:,i:i+1,...], y[:,i:i+1,...], data_range, spatial_dims=spatial_dims)
    pred_rmse = compute_RMSE(pred, y)
    return pred_rmse, pred_psnr, pred_ssim/C

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    eps = 1e-10
    mse_ = compute_MSE(img1, img2)
    if mse_ == 0:
        mse_ += eps
    if torch.is_tensor(img1):
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True, spatial_dims=2):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    # default window_size 11
    if len(img1.size()) == 2:
        shape_ = img1.shape
        img1 = img1.view(1, 1, *shape_)
        img2 = img2.view(1, 1, *shape_)
    window = create_window(window_size, channel, spatial_dims=spatial_dims)
    window = window.type_as(img1)

    conv_op = F.conv2d if spatial_dims == 2 else F.conv3d

    mu1 = conv_op(img1, window, padding=window_size//2)
    mu2 = conv_op(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = conv_op(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = conv_op(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = conv_op(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, spatial_dims=2):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    if spatial_dims == 2:
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    else:
        window = _2D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window
