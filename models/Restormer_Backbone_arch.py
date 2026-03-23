## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from torch.distributed.pipeline.sync.skip.skippable import skippable, stash, pop


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


@skippable(stash=['input'])
class Stash_input_Layer(nn.Module):
    def __init__(self, ):
        super(Stash_input_Layer, self).__init__()

    def forward(self, input):
        yield stash('input', input)
        return input


@skippable(pop=['input'])
class Pop_input_Layer(nn.Module):
    def __init__(self, ):
        super(Pop_input_Layer, self).__init__()

    def forward(self, input):
        pre_input = yield pop('input')
        return input + pre_input


class Sequential_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        self.model = nn.Sequential(
            Stash_input_Layer()
            , LayerNorm(dim, LayerNorm_type)
            , Attention(dim, num_heads, bias)
            , Pop_input_Layer()
            , Stash_input_Layer()
            , LayerNorm(dim, LayerNorm_type)
            , FeedForward(dim, ffn_expansion_factor, bias)
            , Pop_input_Layer()
        )


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim, reduction_factor=8):
        super().__init__()
        self.down_sample_size = input_dim // reduction_factor
        self.activation = nn.ReLU(inplace=True)
        self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, bias=False)
        self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, bias=False)

    def forward(self, x):
        x = self.down_sampler(x)
        x = self.activation(x)
        x = self.up_sampler(x)
        return x


class AdapterTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(AdapterTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.tune_adapter = Adapter(dim, dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x_adapter = self.tune_adapter(x)
        x = x + self.ffn(self.norm2(x))
        x = x + x_adapter
        return x


class FrequnecyAdapterTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, patch_size, ffn_expansion_factor, bias, LayerNorm_type):
        super(FrequnecyAdapterTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.tune_adapter = FrequencyAdapter(dim, dim, patch_size)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x_adapter = self.tune_adapter(x)
        x = x + self.ffn(self.norm2(x))
        x = x + x_adapter
        return x


class FrequencyAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size, reduction_factor=8):
        super(FrequencyAdapter, self).__init__()
        patch_size = int(patch_size)
        self.down_sample_size = input_dim // reduction_factor
        self.activation = nn.ReLU(inplace=True)
        self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, bias=False)
        self.tune_lamb1 = nn.Parameter(torch.zeros((patch_size, patch_size)))
        self.tune_lamb2 = nn.Parameter(torch.zeros((patch_size, patch_size)))
        self.tune_cutoff_frequency = nn.Parameter(torch.tensor(30.))
        self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, bias=False)

    def frequent_decompose(self, x):
        f_transform = torch.fft.fft2(x, dim=(-2, -1))
        f_transform_shifted = torch.fft.fftshift(f_transform)
        rows, cols = x.size(-2), x.size(-1)
        center_row, center_col = rows // 2, cols // 2
        cutoff_frequency = (int)(torch.clamp(self.tune_cutoff_frequency, 0, center_col))
        # 创建低通滤波器
        low_pass_filter = torch.zeros(rows, cols).cuda()
        low_pass_filter[center_row - cutoff_frequency:center_row + cutoff_frequency,
        center_col - cutoff_frequency:center_col + cutoff_frequency] = 1
        # 创建高通滤波器
        high_pass_filter = 1 - low_pass_filter
        # 应用滤波器
        f_transform_shifted_low = f_transform_shifted * low_pass_filter.unsqueeze(0).unsqueeze(0)
        f_transform_shifted_high = f_transform_shifted * high_pass_filter.unsqueeze(0).unsqueeze(0)
        # 逆傅立叶变换
        f_transform_low = torch.fft.ifftshift(f_transform_shifted_low)
        f_transform_high = torch.fft.ifftshift(f_transform_shifted_high)
        x_low = torch.fft.ifft2(f_transform_low, dim=(-2, -1)).abs()
        x_high = torch.fft.ifft2(f_transform_high, dim=(-2, -1)).abs()
        return x_low, x_high

    def forward(self, x):
        x = self.down_sampler(x)
        x = self.activation(x)
        x_low, x_high = self.frequent_decompose(x)
        x = x + self.tune_lamb1 * x_low + self.tune_lamb2 * x_high
        x = self.up_sampler(x)
        return x


##########################################################################
##---------- Restormer -----------------------
class Restormer_Backbone(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=64,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 global_residual=False,
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer_Backbone, self).__init__()
        self.global_residual = global_residual
        print('GR: ', self.global_residual)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def check_image_size(self, x, mode_base=8):
        _, _, h, w = x.size()
        mod_pad_h = (mode_base - h % 8) % mode_base
        mod_pad_w = (mode_base - w % 8) % mode_base
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img):
        H, W = inp_img.shape[2:]

        # print('ori size: ', inp_img.shape)
        inp_img = self.check_image_size(inp_img)
        # print('after size: ', inp_img.shape)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # skip

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # skip

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)  # skip

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            if self.global_residual:
                out_dec_level1 = self.output(out_dec_level1) + inp_img
            else:
                out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1[:, :, :H, :W]


class Restormer_Light_Backbone(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=64,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 global_residual=False,
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer_Light_Backbone, self).__init__()
        self.global_residual = global_residual
        print('GR: ', self.global_residual)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def check_image_size(self, x, mode_base=8):
        _, _, h, w = x.size()
        mod_pad_h = (mode_base - h % 8) % mode_base
        mod_pad_w = (mode_base - w % 8) % mode_base
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img):
        H, W = inp_img.shape[2:]

        # print('ori size: ', inp_img.shape)
        inp_img = self.check_image_size(inp_img)
        # print('after size: ', inp_img.shape)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # C

        inp_enc_level2 = self.down1_2(out_enc_level1)  # 2C
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # 2C

        inp_enc_level3 = self.down2_3(out_enc_level2)  # 4C
        latent = self.latent(inp_enc_level3)  # 4C

        inp_dec_level3 = self.up3_2(latent)  # 2C
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level2], 1)  # 4C
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level3)  # 2C
        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # 2C

        inp_dec_level1 = self.up2_1(out_dec_level2)  # C
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # 2C
        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # 2C

        out_dec_level1 = self.refinement(out_dec_level1)  # 2C

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            if self.global_residual:
                out_dec_level1 = self.output(out_dec_level1) + inp_img
            else:
                out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1[:, :, :H, :W]


class Adapter_Restormer_Backbone(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=64,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 global_residual=False,
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Adapter_Restormer_Backbone, self).__init__()
        self.global_residual = global_residual
        print('GR: ', self.global_residual)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            AdapterTransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                    LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[AdapterTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                                                      ffn_expansion_factor=ffn_expansion_factor,
                                                                      bias=bias, LayerNorm_type=LayerNorm_type) for i in
                                              range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[AdapterTransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                                      ffn_expansion_factor=ffn_expansion_factor,
                                                                      bias=bias, LayerNorm_type=LayerNorm_type) for i in
                                              range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[AdapterTransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                                                              ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                              LayerNorm_type=LayerNorm_type) for i in
                                      range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[AdapterTransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                                      ffn_expansion_factor=ffn_expansion_factor,
                                                                      bias=bias, LayerNorm_type=LayerNorm_type) for i in
                                              range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[AdapterTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                                                      ffn_expansion_factor=ffn_expansion_factor,
                                                                      bias=bias, LayerNorm_type=LayerNorm_type) for i in
                                              range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[AdapterTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                                                      ffn_expansion_factor=ffn_expansion_factor,
                                                                      bias=bias, LayerNorm_type=LayerNorm_type) for i in
                                              range(num_blocks[0])])

        self.refinement = nn.Sequential(*[AdapterTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                                                  ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                                  LayerNorm_type=LayerNorm_type) for i in
                                          range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def check_image_size(self, x, mode_base=8):
        _, _, h, w = x.size()
        mod_pad_h = (mode_base - h % 8) % mode_base
        mod_pad_w = (mode_base - w % 8) % mode_base
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img):
        H, W = inp_img.shape[2:]

        # print('ori size: ', inp_img.shape)
        inp_img = self.check_image_size(inp_img)
        # print('after size: ', inp_img.shape)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            if self.global_residual:
                out_dec_level1 = self.output(out_dec_level1) + inp_img
            else:
                out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1[:, :, :H, :W]


class FrequencyAdapter_Restormer_Backbone(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=64,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 global_residual=False,
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(FrequencyAdapter_Restormer_Backbone, self).__init__()
        self.global_residual = global_residual
        print('GR: ', self.global_residual)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        patch_size = 256

        # 在输入处加一个Prompt，与patch_embed拼接

        self.encoder_level1 = nn.Sequential(*[
            FrequnecyAdapterTransformerBlock(dim=dim, num_heads=heads[0], patch_size=patch_size,
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            FrequnecyAdapterTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], patch_size=patch_size / 2,
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            FrequnecyAdapterTransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                             patch_size=patch_size / (2 ** 2),
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[FrequnecyAdapterTransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                                                                       patch_size=patch_size / (2 ** 3),
                                                                       ffn_expansion_factor=ffn_expansion_factor,
                                                                       bias=bias, LayerNorm_type=LayerNorm_type) for i
                                      in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            FrequnecyAdapterTransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                             patch_size=patch_size / (2 ** 2),
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            FrequnecyAdapterTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                             patch_size=patch_size / (2 ** 1),
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            FrequnecyAdapterTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], patch_size=patch_size,
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            FrequnecyAdapterTransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], patch_size=patch_size,
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def check_image_size(self, x, mode_base=8):
        _, _, h, w = x.size()
        mod_pad_h = (mode_base - h % 8) % mode_base
        mod_pad_w = (mode_base - w % 8) % mode_base
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img):
        H, W = inp_img.shape[2:]

        # print('ori size: ', inp_img.shape)
        inp_img = self.check_image_size(inp_img)
        # print('after size: ', inp_img.shape)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            if self.global_residual:
                out_dec_level1 = self.output(out_dec_level1) + inp_img
            else:
                out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1[:, :, :H, :W]


@skippable(stash=['skip_output1', 'skip_output2', 'skip_output3'])
class Sequential_Transformer_Layer(nn.Module):
    def __init__(self, layerid, dim, head, num_block, ffn_expansion_factor, bias, LayerNorm_type):
        super(Sequential_Transformer_Layer, self).__init__()
        self.layerid = layerid
        self.model = nn.Sequential(*[
            Sequential_TransformerBlock(dim=dim, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                        LayerNorm_type=LayerNorm_type) for i in range(num_block)])

    def forward(self, input):
        output = self.model(input)
        if self.layerid == 1:
            yield stash('skip_output1', output)
        elif self.layerid == 2:
            yield stash('skip_output2', output)
        elif self.layerid == 3:
            yield stash('skip_output3', output)
        return output


@skippable(pop=['skip_output1', 'skip_output2', 'skip_output3'])
class ConcateModule(nn.Module):
    def __init__(self, layerid):
        self.layerid = layerid

    def forward(self, input):
        if self.layerid == 1:
            skip_output = yield pop('skip_output1')
        elif self.layerid == 2:
            skip_output = yield pop('skip_output2')
        elif self.layerid == 3:
            skip_output = yield pop('skip_output3')
        return torch.cat([input, skip_output], 1)


class Restormer_Encoder(nn.Module):
    def __init__(self, inp_channels=3,
                 out_channels=64,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        self.model = nn.Sequential(
            OverlapPatchEmbed(inp_channels, dim)  # patch_embed
            # encoder_level1
            , Sequential_Transformer_Layer(1, dim, heads[0], num_blocks[0], ffn_expansion_factor, bias, LayerNorm_type)
            # encoder_level2
            , Downsample(dim)
            , Sequential_Transformer_Layer(2, int(dim * 2 ** 1), heads[1], num_blocks[1], ffn_expansion_factor, bias,
                                           LayerNorm_type)
            # encoder_level3
            , Downsample(int(dim * 2 ** 1))
            , Sequential_Transformer_Layer(3, int(dim * 2 ** 2), heads[2], num_blocks[2], ffn_expansion_factor, bias,
                                           LayerNorm_type)
            # latent
            , Downsample(int(dim * 2 ** 2))
            , Sequential_Transformer_Layer(0, int(dim * 2 ** 3), heads[3], num_blocks[3], ffn_expansion_factor, bias,
                                           LayerNorm_type)
        )

    def forward(self, input):
        return self.model(input)


class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=64,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        self.model = nn.Sequential(
            # decoder_level3
            Upsample(int(dim * 2 ** 3))
            , ConcateModule(3)
            , nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
            , Sequential_Transformer_Layer(0, int(dim * 2 ** 2), heads[2], num_blocks[2], ffn_expansion_factor, bias,
                                           LayerNorm_type)
            # decoder_level2
            , Upsample(int(dim * 2 ** 2))
            , ConcateModule(2)
            , nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
            , Sequential_Transformer_Layer(0, int(dim * 2 ** 1), heads[1], num_blocks[1], ffn_expansion_factor, bias,
                                           LayerNorm_type)
            # decoder_level1
            , Upsample(int(dim * 2 ** 1))
            , ConcateModule(1)
            , Sequential_Transformer_Layer(0, int(dim * 2 ** 1), heads[0], num_blocks[0], ffn_expansion_factor, bias,
                                           LayerNorm_type)
            # refinement
            , Sequential_Transformer_Layer(0, int(dim * 2 ** 1), heads[0], num_refinement_blocks, ffn_expansion_factor,
                                           bias, LayerNorm_type)
            # output
            , nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, input):
        return self.model(input)

    ########################################################################
    # Model Parallel


class Sinonet_Encoder_Parallel_Restormer_Backbone(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=64,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 global_residual=False,
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 trainer_mode='train'
                 ):

        super(Sinonet_Encoder_Parallel_Restormer_Backbone, self).__init__()
        self.global_residual = global_residual
        self.trainer_mode = trainer_mode
        print('GR: ', self.global_residual)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def check_image_size(self, x, mode_base=8):
        _, _, h, w = x.size()
        mod_pad_h = (mode_base - h % 8) % mode_base
        mod_pad_w = (mode_base - w % 8) % mode_base
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img):
        H, W = inp_img.shape[2:]

        # print('ori size: ', inp_img.shape)
        inp_img = self.check_image_size(inp_img)
        # print('after size: ', inp_img.shape)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # skip

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # skip

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)  # skip

        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)
        if self.trainer_mode == 'train':
            latent = latent.to('cuda:1')
        inp_dec_level3 = self.up4_3(latent)
        if self.trainer_mode == 'train': out_enc_level3 = out_enc_level3.to('cuda:1')
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        if self.trainer_mode == 'train': out_enc_level2 = out_enc_level2.to('cuda:1')
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        if self.trainer_mode == 'train': out_enc_level1 = out_enc_level1.to('cuda:1')
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        if self.trainer_mode == 'train': out_dec_level1 = out_dec_level1.to('cuda:2')
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            if self.global_residual:
                out_dec_level1 = self.output(out_dec_level1) + inp_img
            else:
                out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1[:, :, :H, :W]