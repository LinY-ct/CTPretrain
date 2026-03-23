import torch
import numpy as np
import torch.nn as nn
from torch_radon import Radon, RadonFanbeam
import torch.nn.functional as F
from wrappers.cttools import CTTools
import random
random.seed(123)

"""
The wrappers are used to provide methods for preparing in-completed-view input data.
"""


class FBPTools(nn.Module):
    def __init__(self, img_size=256, num_full_views=720, source_distance=1075, det_count=672):
        super().__init__()
        self.num_full_views = num_full_views
        self.source_distance = source_distance
        self.det_count = det_count
        self.img_size = img_size
        self.cttool = CTTools()

    # ------------ basic radon function ----------------
    # avoid possible cuda error, put radon func in the module
    def get_angles(self, num_views=None, angle_range=np.pi * 2):
        num_views = self.num_full_views if num_views is None else num_views  # specified number of views
        angles = np.linspace(0, angle_range, num_views,
                             endpoint=False)  # select views according to the specified number of views
        return angles

    def radon(self, sinogram, num_views=None, angle_range=np.pi * 2):
        """sinogram to ct image"""
        if angle_range == np.pi * 2:  # full or sparse
            angles = self.get_angles(num_views, np.pi * 2)
        else:  # limited-angle
            angles = self.get_angles(num_views, angle_range)
        radon_tool = RadonFanbeam(self.img_size, angles, self.source_distance, det_count=self.det_count, )
        filter_sin = radon_tool.filter_sinogram(sinogram, "ram-lak")
        back_proj = radon_tool.backprojection(filter_sin)
        return back_proj

    def image_radon(self, ct_image, num_views=None, angle_range=np.pi * 2):
        '''ct image to sinogram'''
        if angle_range == np.pi * 2:  # full or sparse
            angles = self.get_angles(num_views, np.pi * 2)
        else:  # limited-angle
            angles = self.get_angles(None, angle_range)[:num_views]
        radon_tool = RadonFanbeam(self.img_size, angles, self.source_distance, det_count=self.det_count, )
        sinogram = radon_tool.forward(ct_image)
        return sinogram

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward in wrapper should be implemented.')

    # ------------ add poisson + gaussian noisy ----------------
    def add_noise(self, noise_typ, image, mean=0, var=0.1):
        if noise_typ == "gauss":
            # row, col = image.shape
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, image.shape)
            # gauss = gauss.reshape(row, col)
            noisy = image + gauss
            return noisy
        elif noise_typ == "poisson":
            noisy = np.random.poisson(image)
            return noisy

    def generate_noisy_sinogram(self, sino, I0, sigma=0.01):
        "生成泊松噪声+高斯噪声 sinogram"
        bs = sino.shape[0]
        result = []
        for i in range(bs):
            yb = I0 * np.exp(-1.0 * sino[i].cpu().numpy())
            yb = self.add_noise('poisson', yb)
            yb = self.add_noise('gauss', yb, var=sigma)
            yb[yb <= 0.0] = 1e-8
            li_hat = -1.0 * np.log(yb / I0)
            is_nan = np.isnan(li_hat)
            is_inf = np.isinf(li_hat)
            li_hat[is_nan] = 0.0
            li_hat[is_inf] = 0.0
            result.append(li_hat)
        result = np.array(result)
        return torch.from_numpy(result).type(torch.FloatTensor).cuda()

    def generate_sparse_sinogram_mask(self, num_views):
        sinogram_mask = np.arange(1, self.num_full_views + 1)
        if num_views != self.num_full_views:
            sinogram_mask = np.ma.masked_equal(sinogram_mask % (self.num_full_views // num_views), 1)  # 将里面为1的元素标记为掩码值
        else:
            sinogram_mask = np.ma.masked_equal(sinogram_mask % (self.num_full_views // num_views), 0)  # 将里面为0的元素标记为掩码值
        sinogram_mask = sinogram_mask.mask.astype(np.int32)  # [Nv,]
        sinogram_mask = torch.from_numpy(sinogram_mask).float().cuda()  # [Nv,]

        return sinogram_mask

    def geneate_svct_dudo(self, input_sinogram, num_views, return_sinomask=True,mixed_interp=False):

         bs, _, full_num_views, num_det = input_sinogram.shape
         sparse_mask_sinogram_tensor = []
         sparse_CT_mu_tensor = []
         sparse_sinogram_tensor = []
         for b in range(bs):
             sparse_mask_vec = self.generate_sparse_sinogram_mask(num_views)  # [Nv]
             sparse_mask = sparse_mask_vec.reshape(1, 1, len(sparse_mask_vec), 1)  # [1, 1, Nv, 1]
             sparse_mask = sparse_mask.repeat_interleave(num_det, dim=-1)  # [1, 1, Nv, Nd]
             sparse_sinogram_reduce = input_sinogram[b:b + 1, ...].permute(0, 1, 3, 2)[
                 ..., sparse_mask_vec != 0].permute(0, 1, 3, 2).contiguous() #svSinogram
             sparse_CT_mu_tensor.append(self.radon(sparse_sinogram_reduce, num_views=num_views))#svCt
             if mixed_interp:
                 interp_sinogram = F.interpolate(sparse_sinogram_reduce, size=(full_num_views, num_det), mode='bilinear')
                 sparse_sinogram_tensor.append(interp_sinogram)
             else:
                 sparse_sinogram_tensor.append(sparse_mask * input_sinogram[b])
             sparse_mask_sinogram_tensor.append(sparse_mask)

         sparse_mask_sinogram_tensor = torch.cat(sparse_mask_sinogram_tensor, dim=0) # [b, 1, Nv, Nd]
         sparse_CT_mu_tensor = torch.cat(sparse_CT_mu_tensor, dim=0)  # [b, 1, h, w]
         sparse_sinogram_tensor = torch.cat(sparse_sinogram_tensor, dim=0)  # [b, 1, Nv, Nd]
         return sparse_sinogram_tensor, sparse_CT_mu_tensor, sparse_mask_sinogram_tensor

    def generate_lvct_dudo(self, input_sinogram, num_views, return_sinomask=False):
        lv_sinogram = input_sinogram.clone()
        mask_sinogram = torch.zeros_like(lv_sinogram) if return_sinomask else None
        # lvct sampling
        lv_ct_mu_list = []
        for i in range(lv_sinogram.shape[0]):
            view = num_views
            angle_range = view / self.num_full_views * 2 * np.pi
            lv_ct_mu = self.radon(lv_sinogram[i:i + 1, :, :view, :], view, angle_range=angle_range)
            lv_ct_mu_list.append(lv_ct_mu)
            lv_sinogram[i][:,view:,:] = 0.0
            if return_sinomask:
                mask_sinogram[i][:, :view, :] = 1.0
        lv_ct_mu_tensor = torch.cat(lv_ct_mu_list, dim=0)
        return lv_sinogram, lv_ct_mu_tensor, mask_sinogram

    def make_traindata(self, gt_ct_mu, input_ct_mu, degenetaion_type, dose_range, views_range, lv_views_range,
                       return_sinomask=True, mixed_interp=False, return_choice=False):
        gt_sinogram = self.image_radon(gt_ct_mu)
        input_sinogram = self.image_radon(input_ct_mu)
        gt_ct_mu = self.radon(gt_sinogram)

        b, c, fullviews, detectors = gt_sinogram.shape
        mask_sinogram = (self.generate_sparse_sinogram_mask(fullviews)
                         .reshape(1, 1, fullviews, 1)
                         .repeat_interleave(detectors, dim=-1))
        I0, views, lv_views = -1 ,-1,-1
        if degenetaion_type == 'clear':
            return gt_ct_mu, gt_ct_mu, gt_sinogram, gt_sinogram, mask_sinogram
        elif degenetaion_type == 'ld': # 添加噪声
            I0 = random.choice(dose_range)
            input_sinogram = self.generate_noisy_sinogram(gt_sinogram, I0)
            input_ct_mu = self.radon(input_sinogram)

        elif degenetaion_type == 'sv':
            views = random.choice(views_range)
            input_sinogram, input_ct_mu, mask_sinogram = self.geneate_svct_dudo(gt_sinogram, views,
                                                                                return_sinomask=True)
        elif degenetaion_type == 'lv':
            lv_views = random.choice(lv_views_range)
            input_sinogram, input_ct_mu, mask_sinogram = self.generate_lvct_dudo(gt_sinogram, lv_views,
                                                                                  return_sinomask=True)
        elif degenetaion_type == 'ld_sv':
            I0 = random.choice(dose_range)
            views = random.choice(views_range)
            input_sinogram = self.generate_noisy_sinogram(gt_sinogram, I0)
            input_sinogram, input_ct_mu, mask_sinogram = self.geneate_svct_dudo(input_sinogram, views,
                                                                                return_sinomask=True)
        elif degenetaion_type == 'ld_lv':
            I0 = random.choice(dose_range)
            lv_views = random.choice(lv_views_range)
            input_sinogram = self.generate_noisy_sinogram(gt_sinogram, I0)
            input_sinogram, input_ct_mu, mask_sinogram = self.generate_lvct_dudo(input_sinogram, lv_views,
                                                                                 return_sinomask=True)
        elif degenetaion_type == 'ld_mar':
            I0 = random.choice(dose_range)
            input_sinogram = self.generate_noisy_sinogram(input_sinogram, I0)
            input_ct_mu = self.radon(input_sinogram)
        elif degenetaion_type == 'sv_mar':
            views = random.choice(views_range)
            input_sinogram, input_ct_mu, mask_sinogram = self.geneate_svct_dudo(input_sinogram, views,
                                                                                return_sinomask=True)
        elif degenetaion_type == 'lv_mar':
            lv_views = random.choice(lv_views_range)
            input_sinogram, input_ct_mu, mask_sinogram = self.generate_lvct_dudo(input_sinogram, lv_views,
                                                                                 return_sinomask=True)
        elif degenetaion_type == 'ld_sv_mar':
            I0 = random.choice(dose_range)
            input_sinogram = self.generate_noisy_sinogram(input_sinogram, I0)
            views = random.choice(views_range)
            input_sinogram, input_ct_mu, mask_sinogram = self.geneate_svct_dudo(input_sinogram, views,
                                                                                return_sinomask=True)
        elif degenetaion_type == 'ld_lv_mar':
            I0 = random.choice(dose_range)
            input_sinogram = self.generate_noisy_sinogram(input_sinogram, I0)
            lv_views = random.choice(lv_views_range)
            input_sinogram, input_ct_mu, mask_sinogram = self.generate_lvct_dudo(input_sinogram, lv_views,
                                                                                 return_sinomask=True)
        else : # mar
            pass

        # print('degenetaion_type I0, views, lv_views', degenetaion_type, I0, views, lv_views)
        if return_choice:
            return gt_ct_mu, input_ct_mu, gt_sinogram, input_sinogram, mask_sinogram, degenetaion_type,str(I0)+','+str(views)+','+str(lv_views)
        return gt_ct_mu, input_ct_mu, gt_sinogram, input_sinogram, mask_sinogram

    def make_traindata2(self, gt_ct_mu, input_ct_mu, degenetaion_type, dose, view,
                       return_sinomask=True, mixed_interp=False, return_choice=False):
        gt_sinogram = self.image_radon(gt_ct_mu)
        input_sinogram = self.image_radon(input_ct_mu)
        gt_ct_mu = self.radon(gt_sinogram)

        b, c, fullviews, detectors = gt_sinogram.shape
        mask_sinogram = (self.generate_sparse_sinogram_mask(fullviews)
                         .reshape(1, 1, fullviews, 1)
                         .repeat_interleave(detectors, dim=-1))
        # I0, views, lv_views = -1 ,-1,-1
        if degenetaion_type == 'clear':
            return gt_ct_mu, gt_ct_mu, gt_sinogram, gt_sinogram, mask_sinogram
        elif degenetaion_type == 'ld': # 添加噪声
            input_sinogram = self.generate_noisy_sinogram(gt_sinogram, dose)
            input_ct_mu = self.radon(input_sinogram)

        elif degenetaion_type == 'sv':
            input_sinogram, input_ct_mu, mask_sinogram = self.geneate_svct_dudo(gt_sinogram, view,
                                                                                return_sinomask=True)
        elif degenetaion_type == 'lv':
            input_sinogram, input_ct_mu, mask_sinogram = self.generate_lvct_dudo(gt_sinogram, view,
                                                                                  return_sinomask=True)
        elif degenetaion_type == 'ld_sv':
            input_sinogram = self.generate_noisy_sinogram(gt_sinogram, dose)
            input_sinogram, input_ct_mu, mask_sinogram = self.geneate_svct_dudo(input_sinogram, view,
                                                                                return_sinomask=True)
        elif degenetaion_type == 'ld_lv':


            input_sinogram = self.generate_noisy_sinogram(gt_sinogram, dose)
            input_sinogram, input_ct_mu, mask_sinogram = self.generate_lvct_dudo(input_sinogram, view,
                                                                                 return_sinomask=True)
        elif degenetaion_type == 'ld_mar':

            input_sinogram = self.generate_noisy_sinogram(input_sinogram, dose)
            input_ct_mu = self.radon(input_sinogram)
        elif degenetaion_type == 'sv_mar':

            input_sinogram, input_ct_mu, mask_sinogram = self.geneate_svct_dudo(input_sinogram, view,
                                                                                return_sinomask=True)
        elif degenetaion_type == 'lv_mar':

            input_sinogram, input_ct_mu, mask_sinogram = self.generate_lvct_dudo(input_sinogram, view,
                                                                                 return_sinomask=True)
        elif degenetaion_type == 'ld_sv_mar':

            input_sinogram = self.generate_noisy_sinogram(input_sinogram, dose)
            input_sinogram, input_ct_mu, mask_sinogram = self.geneate_svct_dudo(input_sinogram, view,
                                                                                return_sinomask=True)
        elif degenetaion_type == 'ld_lv_mar':
            input_sinogram = self.generate_noisy_sinogram(input_sinogram, dose)
            input_sinogram, input_ct_mu, mask_sinogram = self.generate_lvct_dudo(input_sinogram, view,
                                                                                 return_sinomask=True)
        else : # mar
            raise NotImplementedError(f"not support degenetaion_type:{degenetaion_type} ")

        # print('degenetaion_type I0, views, lv_views', degenetaion_type, I0, views, lv_views)
        if return_choice:
            return gt_ct_mu, input_ct_mu, gt_sinogram, input_sinogram, mask_sinogram, degenetaion_type,str(dose)+','+str(view)
        return gt_ct_mu, input_ct_mu, gt_sinogram, input_sinogram, mask_sinogram


