import torch
import torch.nn as nn
from torchvision.models import vgg16
import torchvision
import torch.nn.functional as F

#
# class ContrastLoss(nn.Module):
#     def __init__(self, isFreq=False):
#         super(ContrastLoss, self).__init__()
#         self.isFreq = isFreq
#         self.l1 = nn.L1Loss()
#         self.model = vgg16(pretrained=True)
#         self.model = self.model.features[:16].to("cuda" if torch.cuda.is_available() else "cpu")
#         for param in self.model.parameters():
#             param.requires_grad = False
#         self.layer_name_mapping = {
#             '3': "relu1_2",
#             '8': "relu2_2",
#             '15': "relu3_3"
#         }
#         self.rate_conv1 = nn.Sequential(
#             nn.Conv2d(64, 64 // 8, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(64 // 8, 2, 1, bias=False),
#         )
#         self.rate_conv2 = nn.Sequential(
#             nn.Conv2d(128, 128 // 8, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(128 // 8, 2, 1, bias=False),
#         )
#         self.rate_conv3 = nn.Sequential(
#             nn.Conv2d(256, 256 // 8, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(256 // 8, 2, 1, bias=False),
#         )
#         self.rate_conv = [self.rate_conv1, self.rate_conv2, self.rate_conv3]
#     def shift(self, x):
#         '''shift FFT feature map to center'''
#         b, c, h, w = x.shape
#         return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))
#
#     def unshift(self, x):
#         """converse to shift operation"""
#         b, c, h, w = x.shape
#         return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))
#
#     def fft(self, x, index, n=128):
#         """obtain high/low-frequency features from input"""
#         # x = self.conv1(x)
#         mask = torch.zeros(x.shape).to(x.device)
#         h, w = x.shape[-2:]
#         threshold = F.adaptive_avg_pool2d(x, 1)
#         threshold = self.rate_conv[index](threshold).sigmoid()
#
#         for i in range(mask.shape[0]):
#             h_ = (h // n * threshold[i, 0, :, :]).int()
#             w_ = (w // n * threshold[i, 1, :, :]).int()
#
#             mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1
#
#         fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
#         fft = self.shift(fft)
#
#         fft_high = fft * (1 - mask)
#
#         high = self.unshift(fft_high)
#         high = torch.fft.ifft2(high, norm='forward', dim=(-2, -1))
#         high = torch.abs(high)
#
#         # fft_low = fft * mask
#
#         # low = self.unshift(fft_low)
#         # low = torch.fft.ifft2(low, norm='forward', dim=(-2, -1))
#         # low = torch.abs(low)
#
#         return high#, low
#     def gen_features(self, x):
#         output = []
#         for name, module in self.model._modules.items():
#             x = module(x)
#             if name in self.layer_name_mapping:
#                 output.append(x)
#         if self.isFreq :
#             for i in range(len(output)):
#                 output[i] = self.fft(output[i],i)
#         return output
#     def forward(self, inp, pos, neg, out):
#         inp_t = inp
#         inp_x0 = self.gen_features(inp_t)
#         pos_t = pos
#         pos_x0 = self.gen_features(pos_t)
#         out_t = out
#         out_x0 = self.gen_features(out_t)
#         neg_t, neg_x0 = [],[]
#         for i in range(neg.shape[1]):
#             neg_i = neg[:,i,:,:]
#             neg_t.append(neg_i)
#             neg_x0_i = self.gen_features(neg_i)
#             neg_x0.append(neg_x0_i)
#         loss = 0
#         for i in range(len(pos_x0)):
#             pos_term = self.l1(out_x0[i], pos_x0[i].detach())
#             inp_term = self.l1(out_x0[i], inp_x0[i].detach())/(len(neg_x0)+1)
#             neg_term = sum(self.l1(out_x0[i], neg_x0[j][i].detach()) for j in range(len(neg_x0)))/(len(neg_x0)+1)
#             loss = loss + pos_term / (inp_term+neg_term+1e-7)
#         return loss / len(pos_x0)


class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.model = vgg16(pretrained=True)
        self.model = self.model.features[:16].to("cuda" if torch.cuda.is_available() else "cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def gen_features(self, x):
        output = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output
    def forward(self, inp, pos, neg, out):
        inp_t = inp
        inp_x0 = self.gen_features(inp_t)
        pos_t = pos
        pos_x0 = self.gen_features(pos_t)
        out_t = out
        out_x0 = self.gen_features(out_t)
        neg_t, neg_x0 = [],[]
        for i in range(neg.shape[1]):
            neg_i = neg[:,i,:,:]
            neg_t.append(neg_i)
            neg_x0_i = self.gen_features(neg_i)
            neg_x0.append(neg_x0_i)
        loss = 0
        for i in range(len(pos_x0)):
            pos_term = self.l1(out_x0[i], pos_x0[i].detach())
            inp_term = self.l1(out_x0[i], inp_x0[i].detach())/(len(neg_x0)+1)
            neg_term = sum(self.l1(out_x0[i], neg_x0[j][i].detach()) for j in range(len(neg_x0)))/(len(neg_x0)+1)
            loss = loss + pos_term / (inp_term+neg_term+1e-7)
        return loss / len(pos_x0)




class GGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.center = nn.Parameter(torch.tensor([0.], dtype=torch.float32), requires_grad=True)
        self.width = nn.Parameter(torch.tensor([1.], dtype=torch.float32), requires_grad=True)
    def gen_gaussian_bandpass(self, center=0, width=0.2, shape=(256, 129),
                              shift_to_fit_fft=True, unsqueeze=True, threshold=False):
        # accept input format: [C, 1]
        def _make_shape(x):
            if not torch.is_tensor(x):
                if not isinstance(x, (tuple, list)):
                    x = torch.tensor([x])
                else:
                    x = torch.tensor(x)
            if x.ndim <= 3:
                dim = x.numel()
                x = x.reshape(dim, 1, 1)
            return x

        center = _make_shape(center).clamp(0, 1)
        width = _make_shape(width).clamp(min=1e-12, max=2)
        assert center.shape == width.shape, f'center: {center} != width: {width}.'
        X, Y = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
        X = torch.repeat_interleave(X.unsqueeze(0), center.shape[0], dim=0).to(center.device)
        Y = torch.repeat_interleave(Y.unsqueeze(0), center.shape[0], dim=0).to(center.device)
        x0 = (shape[0] - 1) // 2
        y0 = 0
        D2 = ((X - x0) ** 2 + (Y - y0) ** 2).float()
        D2 /= D2.max()
        H = torch.exp(-((D2 - center ** 2) / (D2.sqrt() * width + 1e-12)) ** 2)
        H = torch.roll(H, H.shape[-2] // 2 + 1, -2) if shift_to_fit_fft else H
        H = H.unsqueeze(0) if unsqueeze else H
        if threshold:
            H_mean = H.mean()
            H[H < H_mean] = 0.0
            H[H >= H_mean] = 1.0
        return H
    def forward(self, x):
        mask = self.gen_gaussian_bandpass(self.center, self.width, shape=x.shape[2:])
        return mask * x


class GGBCL(nn.Module):
    def __init__(self):
        super(GGBCL, self).__init__()
        self.l1 = nn.L1Loss()
        self.model = vgg16(pretrained=True)
        self.model = self.model.features[:16].to("cuda" if torch.cuda.is_available() else "cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        self.fu1 = GGB()
        self.fu2 = GGB()
        self.fu3 = GGB()
        self.fu = [self.fu1, self.fu2, self.fu3]

    def gen_features(self, x):
        output = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        for i in range(len(output)):
            output[i] = self.fu[i](output[i])
        return output

    def forward(self, inp, pos, neg, out):
        inp_t = inp
        inp_x0 = self.gen_features(inp_t)
        pos_t = pos
        pos_x0 = self.gen_features(pos_t)
        out_t = out
        out_x0 = self.gen_features(out_t)
        neg_t, neg_x0 = [],[]
        for i in range(neg.shape[1]):
            neg_i = neg[:,i,:,:]
            neg_t.append(neg_i)
            neg_x0_i = self.gen_features(neg_i)
            neg_x0.append(neg_x0_i)
        loss = 0
        for i in range(len(pos_x0)):
            pos_term = self.l1(out_x0[i], pos_x0[i].detach())
            inp_term = self.l1(out_x0[i], inp_x0[i].detach())/(len(neg_x0)+1)
            neg_term = sum(self.l1(out_x0[i], neg_x0[j][i].detach()) for j in range(len(neg_x0)))/(len(neg_x0)+1)
            loss = loss + pos_term / (inp_term+neg_term+1e-7)
        return loss / len(pos_x0)


class GGB3(nn.Module):
    def __init__(self):
        super().__init__()
        self.center = nn.Parameter(torch.tensor([0.], dtype=torch.float32), requires_grad=True)
        self.width = nn.Parameter(torch.tensor([1.], dtype=torch.float32), requires_grad=True)
    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def gen_gaussian_bandpass(self, center=0, width=0.2, shape=(256, 129),
                              shift_to_fit_fft=True, unsqueeze=True, threshold=False):
        # accept input format: [C, 1]
        def _make_shape(x):
            if not torch.is_tensor(x):
                if not isinstance(x, (tuple, list)):
                    x = torch.tensor([x])
                else:
                    x = torch.tensor(x)
            if x.ndim <= 3:
                dim = x.numel()
                x = x.reshape(dim, 1, 1)
            return x

        center = _make_shape(center).clamp(0, 1)
        width = _make_shape(width).clamp(min=1e-12, max=2)
        assert center.shape == width.shape, f'center: {center} != width: {width}.'
        X, Y = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
        X = torch.repeat_interleave(X.unsqueeze(0), center.shape[0], dim=0).to(center.device)
        Y = torch.repeat_interleave(Y.unsqueeze(0), center.shape[0], dim=0).to(center.device)
        x0 = (shape[0] - 1) // 2
        y0 = 0
        D2 = ((X - x0) ** 2 + (Y - y0) ** 2).float()
        D2 /= D2.max()
        H = torch.exp(-((D2 - center ** 2) / (D2.sqrt() * width + 1e-12)) ** 2)
        H = torch.roll(H, H.shape[-2] // 2 + 1, -2) if shift_to_fit_fft else H
        H = H.unsqueeze(0) if unsqueeze else H
        if threshold:
            H_mean = H.mean()
            H[H < H_mean] = 0.0
            H[H >= H_mean] = 1.0
        return H
    def forward(self, x):
        fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
        # fft = self.shift(fft)
        mask = self.gen_gaussian_bandpass(self.center, self.width, shape=x.shape[2:])
        fft_high = fft * mask
        # high = self.unshift(fft_high)
        high = torch.fft.ifft2(fft_high, norm='forward', dim=(-2, -1))
        high = torch.abs(high)

        return high


class GGB3CL(nn.Module):
    def __init__(self):
        super(GGB3CL, self).__init__()
        self.l1 = nn.L1Loss()
        self.model = vgg16(pretrained=True)
        self.model = self.model.features[:16].to("cuda" if torch.cuda.is_available() else "cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        self.fu1 = GGB3()
        self.fu2 = GGB3()
        self.fu3 = GGB3()
        self.fu = [self.fu1, self.fu2, self.fu3]

    def gen_features(self, x):
        output = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        for i in range(len(output)):
            output[i] = self.fu[i](output[i])
        return output

    def forward(self, inp, pos, neg, out):
        inp_t = inp
        inp_x0 = self.gen_features(inp_t)
        pos_t = pos
        pos_x0 = self.gen_features(pos_t)
        out_t = out
        out_x0 = self.gen_features(out_t)
        neg_t, neg_x0 = [],[]
        for i in range(neg.shape[1]):
            neg_i = neg[:,i,:,:]
            neg_t.append(neg_i)
            neg_x0_i = self.gen_features(neg_i)
            neg_x0.append(neg_x0_i)
        loss = 0
        for i in range(len(pos_x0)):
            pos_term = self.l1(out_x0[i], pos_x0[i].detach())
            inp_term = self.l1(out_x0[i], inp_x0[i].detach())/(len(neg_x0)+1)
            neg_term = sum(self.l1(out_x0[i], neg_x0[j][i].detach()) for j in range(len(neg_x0)))/(len(neg_x0)+1)
            loss = loss + pos_term / (inp_term+neg_term+1e-7)
        return loss / len(pos_x0)

class GGB3CL_v2(nn.Module):
    def __init__(self):
        super(GGB3CL_v2, self).__init__()
        self.l1 = nn.L1Loss()
        self.model = vgg16(pretrained=True)
        self.model = self.model.features[:16].to("cuda" if torch.cuda.is_available() else "cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def gen_features(self, x):
        output = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output
    def forward(self, mask, inp, pos, neg, out):

        inp_fft = torch.fft.fft2(inp, norm='forward', dim=(-2, -1))
        inp_fft_high = inp_fft * mask
        inp_high = torch.fft.ifft2(inp_fft_high, norm='forward', dim=(-2, -1))
        inp = torch.abs(inp_high)

        pos_fft = torch.fft.fft2(pos, norm='forward', dim=(-2, -1))
        pos_fft_high = pos_fft * mask
        pos_high = torch.fft.ifft2(pos_fft_high, norm='forward', dim=(-2, -1))
        pos = torch.abs(pos_high)

        neg_fft = torch.fft.fft2(neg, norm='forward', dim=(-2, -1))
        neg_fft_high = neg_fft * mask
        neg_high = torch.fft.ifft2(neg_fft_high, norm='forward', dim=(-2, -1))
        neg = torch.abs(neg_high)

        out_fft = torch.fft.fft2(out, norm='forward', dim=(-2, -1))
        out_fft_high = out_fft * mask
        out_high = torch.fft.ifft2(out_fft_high, norm='forward', dim=(-2, -1))
        out = torch.abs(out_high)

        inp_t = inp
        inp_x0 = self.gen_features(inp_t)
        pos_t = pos
        pos_x0 = self.gen_features(pos_t)
        out_t = out
        out_x0 = self.gen_features(out_t)
        neg_t, neg_x0 = [],[]
        for i in range(neg.shape[1]):
            neg_i = neg[:,i,:,:]
            neg_t.append(neg_i)
            neg_x0_i = self.gen_features(neg_i)
            neg_x0.append(neg_x0_i)
        loss = 0
        for i in range(len(pos_x0)):
            pos_term = self.l1(out_x0[i], pos_x0[i].detach())
            inp_term = self.l1(out_x0[i], inp_x0[i].detach())/(len(neg_x0)+1)
            neg_term = sum(self.l1(out_x0[i], neg_x0[j][i].detach()) for j in range(len(neg_x0)))/(len(neg_x0)+1)
            loss = loss + pos_term / (inp_term+neg_term+1e-7)
        return loss / len(pos_x0)