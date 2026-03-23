import torch
import torch.nn as nn
from torchvision.models import resnet34
from timm.models.layers import DropPath
from models.V2.activation import mish
import torch.nn.functional as F


class Mlp(nn.Module):
    "Implementation of MLP"

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim=512, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 使用 3D 卷积替换 2D 卷积
        self.sr1 = nn.Conv3d(int(dim / 4), int(dim / 4), kernel_size=(1, 1, 1), stride=2)
        self.sr2 = nn.Conv3d(int(dim / 4), int(dim / 4), kernel_size=(3, 3, 3), stride=sr_ratio, padding=1, dilation=1)
        self.sr3 = nn.Conv3d(int(dim / 4), int(dim / 4), kernel_size=(3, 3, 3), stride=sr_ratio, padding=2, dilation=2)
        self.sr4 = nn.Conv3d(int(dim / 4), int(dim / 4), kernel_size=(3, 3, 3), stride=sr_ratio, padding=3, dilation=3)

        self.pos1 = PosCNN(int(dim / 4), int(dim / 4))
        self.pos2 = PosCNN(int(dim / 4), int(dim / 4))
        self.pos3 = PosCNN(int(dim / 4), int(dim / 4))
        self.pos4 = PosCNN(int(dim / 4), int(dim / 4))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape  # 新的输入形状: [B, D, H, W, C]
        D = 6
        H = W = int((N / D) ** (1 / 2))  # 新的展平后的大小

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # print('x:',x.shape)
        # print('q:',x.shape)
        # 处理3D张量
        x = x.reshape(B, C, D, H, W)
        # print(x.shape)
        x1 = x[:, :int(C / 4), :, :, :]
        x2 = x[:, int(C / 4):int(C / 4) * 2, :, :, :]
        x3 = x[:, int(C / 4) * 2:int(C / 4) * 3, :, :, :]
        x4 = x[:, int(C / 4) * 3:int(C / 4) * 4, :, :, :]

        # print('x1:',x1.shape)
        sr1 = self.sr1(x1).reshape(B, int(C / 4), -1).permute(0, 2, 1)
        sr2 = self.sr2(x2).reshape(B, int(C / 4), -1).permute(0, 2, 1)
        sr3 = self.sr3(x3).reshape(B, int(C / 4), -1).permute(0, 2, 1)
        sr4 = self.sr4(x4).reshape(B, int(C / 4), -1).permute(0, 2, 1)

        # print('self.sr4:',self.sr4(x4).shape)
        # print('sr4:',sr4.shape)
        sr_d = int(D / 2)
        sr_h = sr_w = int((sr4.shape[1] / sr_d) ** (1 / 2))
        # print('sr_d,sr_h,sr_w:',sr_d,sr_h,sr_w)
        sr1 = self.pos1(sr1, sr_d, sr_h, sr_w)
        sr2 = self.pos2(sr2, sr_d, sr_h, sr_w)
        sr3 = self.pos3(sr3, sr_d, sr_h, sr_w)
        sr4 = self.pos4(sr4, sr_d, sr_h, sr_w)

        # 将四个部分合并
        x_sr = torch.cat((sr1, sr2, sr3, sr4), dim=2)
        x_sr = self.norm(x_sr)

        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        # 使用 3D 卷积
        self.proj = nn.Conv3d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim)
        self.s = s

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        feat_token = x
        # 重塑为 3D 特征图 [B, C, D, H, W]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, D, H, W)

        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat  # 残差连接
        else:
            x = self.proj(cnn_feat)

        # 拉平最后的 D, H, W 维度，并转置
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    Implementation of Transformer,
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, f_kv, f3):
        x0 = f3
        x_b, x_c, x_h, x_w = x.shape
        x = x.flatten(2).transpose(2, 1)
        f_kv = f_kv.flatten(2).transpose(2, 1)
        B, N, C = x.shape
        # print(x.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(f_kv).reshape(B, N, 2, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        #         print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(2, 1).reshape(x_b, x_c, x_h, x_w)
        out = out + x0

        return out


def l2_normalization(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x


class ResNet34Mvt(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 attn_drop=0.,
                 drop_path_rate=0.1, drop_rate=0., ):
        super(ResNet34Mvt, self).__init__()

        # 三个不同卷积核大小的二维卷积层
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels=32, kernel_size=1, padding=0), mish())
        self.bn1 = nn.Sequential(nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv2 = nn.Sequential(nn.Conv2d(32, out_channels=32, kernel_size=3, padding=1), mish())
        self.bn2 = nn.Sequential(nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv3 = nn.Sequential(nn.Conv2d(32, out_channels=32, kernel_size=5, padding=2), mish())
        self.bn3 = nn.Sequential(nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv4 = nn.Sequential(nn.Conv2d(32, out_channels=32, kernel_size=7, padding=3), mish())
        self.bn4 = nn.Sequential(nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv5 = nn.Sequential(nn.Conv2d(32, out_channels=32, kernel_size=5, padding=2), mish())
        self.bn5 = nn.Sequential(nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv6 = nn.Sequential(nn.Conv2d(32, out_channels=32, kernel_size=3, padding=1), mish())
        self.bn6 = nn.Sequential(nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True), mish())

        self.transformer = TransformerBlock(
            dim=32, num_heads=2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop_path=drop_path_rate)

        self.conv7 = nn.Sequential(nn.Conv3d(32, out_channels, kernel_size=(6, 1, 1), stride=(1, 1, 1), padding=0),
                                   mish())
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)


    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        f1 = self.bn1(self.conv1(x))
        f2 = self.bn2(self.conv2(f1))
        f3 = self.bn3(self.conv3(f2))
        f4 = self.bn4(self.conv4(f3))
        f5 = self.bn5(self.conv5(f4))
        f6 = self.bn6(self.conv6(f5))
        x = torch.cat(
            (f1.unsqueeze(2), f2.unsqueeze(2), f3.unsqueeze(2), f4.unsqueeze(2), f5.unsqueeze(2), f6.unsqueeze(2)),
            dim=2)

        f_b, f_c, f_d, f_h, f_w = x.shape

        # print('x shape:', x.shape)

        f = x.flatten(2).transpose(1, 2)

        # print(f.shape)
        f1 = self.transformer(f)
        f1 = f1.transpose(1, 2)
        f1 = f1.reshape(f_b, f_c, f_d, f_h, f_w)
        f1 = f1 + x
        f1 = self.conv7(f1).squeeze(2)
        f1 = self.upsample(f1)
        # print('out:',f1.shape)

        return f1

