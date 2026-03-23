# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from models import common
from wrappers.FBPtools import FBPTools
import math
import torch
import torch.nn.functional as F
import torch.nn.parallel as P

from torch import nn, Tensor
from einops import rearrange
import copy


def make_model(args, parent=False):
    return ipt(args)


class IPT(FBPTools):
    def __init__(self,n_feats=64, shift_mean=True, precision='single',
                        rgb_range=255, scale=[1], n_colors=3, patch_size=48, patch_dim=3,
                        num_heads=12, num_layers=12, num_queries=6, dropout_rate=0,
                        no_mlp=False, pos_every=False, no_pos=False, no_norm=False,
                        conv=common.default_conv,
                        crop_batch_size=8 ):
        super().__init__()
        self.crop_batch_size = crop_batch_size
        self.patch_size = patch_size
        self.scale = scale
        self.idx_scale = 0
        self.n_GPUs = 1
        self.slice_size = patch_size
        self.overlap_size = 12
        self.model = ipt(n_feats=n_feats, shift_mean=shift_mean, precision=precision,
                         rgb_range=rgb_range, scale=scale, n_colors=n_colors, patch_size=patch_size,
                         patch_dim=patch_dim,
                         num_heads=num_heads, num_layers=num_layers, num_queries=num_queries, dropout_rate=dropout_rate,
                         no_mlp=no_mlp, pos_every=pos_every, no_pos=no_pos, no_norm=no_norm, conv=conv,
                         crop_batch_size=crop_batch_size).cuda()
    def tune(self,):
        net_checkpath = "/mnt/nas/wsy/Codes/Pretrained-IPT/pretrainmodel/IPT_pretrain.pt"
        net_checkpoint = torch.load(net_checkpath, map_location='cpu')
        # print(net_checkpoint.keys())
        # print(self.model.state_dict().keys())
        model_dict = self.model.state_dict()
        # 从预训练参数中去掉不匹配的参数
        pretrained_dict = {k: v for k, v in net_checkpoint.items() if k in model_dict}
        # 更新当前模型的参数字典
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=True)
        print('Loading model for IPT [{:s}] ...'.format(net_checkpath))
    def imageSlicing(self, x):
        batch_size, channels, height, width = x.size()

        # 计算水平和垂直方向上的切片数量
        num_horizontal_slices = (width - self.slice_size) // (self.slice_size - self.overlap_size) + 2
        num_vertical_slices = (height - self.slice_size) // (self.slice_size - self.overlap_size) + 2
        # 初始化一个张量来存储所有的切片
        slices = torch.zeros(batch_size, channels, num_horizontal_slices * num_vertical_slices, self.slice_size,
                             self.slice_size).cuda()
        # 切片图像
        for i in range(num_horizontal_slices):
            for j in range(num_vertical_slices):
                if i != num_horizontal_slices - 1 and j != num_vertical_slices - 1:
                    # 计算切片的位置
                    left = i * (self.slice_size - self.overlap_size)
                    upper = j * (self.slice_size - self.overlap_size)
                    right = min(left + self.slice_size, width)
                    lower = min(upper + self.slice_size, height)
                elif i == num_horizontal_slices - 1 and j == num_vertical_slices - 1:
                    left = width - self.slice_size
                    right = width
                    upper = height - self.slice_size
                    lower = height
                elif i == num_horizontal_slices - 1:
                    left = width - self.slice_size
                    right = width
                    upper = j * (self.slice_size - self.overlap_size)
                    lower = upper + self.slice_size
                elif j == num_vertical_slices - 1:
                    left = i * (self.slice_size - self.overlap_size)
                    right = left + self.slice_size
                    upper = height - self.slice_size
                    lower = height

                # 切片图像
                slice_img = x[:, :, upper:lower, left:right]
                slices[:, :, j * num_horizontal_slices + i, :slice_img.size(2), :slice_img.size(3)] = slice_img
        #         b, c, n, h, w = slices.size

        return slices  # .view(b*n, c, h, w)

    def image_concatenate(self, slices, image_shape, overlap_size):
        batch_size_num_slices, channels, num_slices, slice_height, slice_width = slices.size()
        batch_size, channels, height, width = image_shape
        num_horizontal_slices = (width - slice_width) // (slice_width - overlap_size) + 2
        num_vertical_slices = (height - slice_height) // (slice_height - overlap_size) + 2
        # 初始化一个张量来存储拼接后的图像
        reconstructed_image = torch.zeros(batch_size, channels, height, width).cuda()
        # 初始化一个张量来记录每个像素被叠加的次数
        overlap_counter = torch.zeros(batch_size, 1, height, width).cuda()

        # 拼接图像
        for i in range(num_horizontal_slices):
            for j in range(num_vertical_slices):
                if i != num_horizontal_slices - 1 and j != num_vertical_slices - 1:
                    # 计算切片的位置
                    left = i * (slice_width - overlap_size)
                    upper = j * (slice_height - overlap_size)
                    right = min(left + slice_width, width)
                    lower = min(upper + slice_height, height)
                elif i == num_horizontal_slices - 1 and j == num_vertical_slices - 1:
                    left = width - slice_width
                    upper = height - slice_height
                    right = width
                    lower = height
                elif i == num_horizontal_slices - 1:
                    left = width - slice_width
                    upper = j * (slice_height - overlap_size)
                    right = width
                    lower = upper + slice_height
                elif j == num_vertical_slices - 1:
                    left = i * (slice_width - overlap_size)
                    upper = height - slice_height
                    right = left + slice_width
                    lower = height

                # 获取切片的索引
                slice_index = j * num_horizontal_slices + i

                # 在拼接时，考虑到重叠区域，进行叠加
                #                 print('='*20)
                #                 print('i:', i, ' j:',j)
                #                 print( upper,lower)
                #                 print(left,right)
                #                 print(reconstructed_image[:, :, upper:lower, left:right].shape)
                #                 print(slices[:, :, slice_index, :, :].shape)
                #                 print('='*20)
                reconstructed_image[:, :, upper:lower, left:right] += slices[:, :, slice_index, :, :]
                overlap_counter[:, :, upper:lower, left:right] += 1

        # 根据叠加次数，对拼接后的图像进行平均处理
        reconstructed_image /= torch.clamp(overlap_counter, min=1)  # 防止除以0的情况

        return reconstructed_image

    def forward_chop(self, x, shave=12):
        patchs = self.imageSlicing(x)  # b, c, n, ps,ps
        b, c, n, ps, ps = patchs.shape
        patchs = patchs.view(b * n, c, ps, ps)
        output_patchs = self.model.forward(patchs)
        output_patchs = output_patchs.view(b, c, n, ps, ps)
        output = self.image_concatenate(output_patchs, x.shape, self.overlap_size)
        return output
        # n = b*n // self.crop_batch_size

        # for i in range(1,n+1):
        #     x_chop = patchs[:i*self.crop_batch_size, ]
        #     P.data_parallel(self.model, *x, range(self.n_GPUs))

        # x.cpu() # 1, 3, 321, 418
        # batchsize = self.crop_batch_size # 64
        # h, w = x.size()[-2:]
        # padsize = int(self.patch_size) # 48
        # shave = int(self.patch_size/2) # 24

        # scale = self.scale[self.idx_scale]

        # h_cut = (h-padsize)%(int(shave/2)) # 9
        # w_cut = (w-padsize)%(int(shave/2)) # 1

        # x_unfold = torch.nn.functional.unfold(x, padsize, stride=int(shave/2)).transpose(0,2).contiguous() # torch.Size([851, 6912, 1])

        # x_hw_cut = x[...,(h-padsize):,(w-padsize):] # torch.Size([1, 3, 48, 48])
        # y_hw_cut = self.model.forward(x_hw_cut.cuda()).cpu() # torch.Size([1, 3, 48, 48])

        # x_h_cut = x[...,(h-padsize):,:] # torch.Size([1, 3, 48, 481])
        # x_w_cut = x[...,:,(w-padsize):] # torch.Size([1, 3, 321, 48])
        # y_h_cut = self.cut_h(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize) # torch.Size([1, 3, 48, 480])
        # y_w_cut = self.cut_w(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize) # torch.Size([1, 3, 312, 48])

        # x_h_top = x[...,:padsize,:] #torch.Size([1, 3, 48, 481])
        # x_w_top = x[...,:,:padsize] #torch.Size([1, 3, 321, 48])
        # y_h_top = self.cut_h(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize) # torch.Size([1, 3, 48, 480])
        # y_w_top = self.cut_w(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize) # torch.Size([1, 3, 312, 48])

        # x_unfold = x_unfold.view(x_unfold.size(0),-1,padsize,padsize) #torch.Size([851, 3, 48, 48])
        # y_unfold = []

        # x_range = x_unfold.size(0)//batchsize + (x_unfold.size(0)%batchsize !=0) #14
        # x_unfold.cuda()
        # for i in range(x_range):
        #     y_unfold.append(self.model.forward(x_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())
        #     # y_unfold.append(P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu()) #torch.Size([64, 3, 48, 48])
        # y_unfold = torch.cat(y_unfold,dim=0) #torch.Size([851, 3, 48, 48])

        # y = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut)*scale,(w-w_cut)*scale), padsize*scale, stride=int(shave/2*scale))# torch.Size([1, 3, 312, 480])

        # y[...,:padsize*scale,:] = y_h_top
        # y[...,:,:padsize*scale] = y_w_top

        # y_unfold = y_unfold[...,int(shave/2*scale):padsize*scale-int(shave/2*scale),int(shave/2*scale):padsize*scale-int(shave/2*scale)].contiguous() #torch.Size([851, 3, 24, 24])
        # y_inter = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), padsize*scale-shave*scale, stride=int(shave/2*scale))#torch.Size([1, 3, 288, 456])

        # y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype) # torch.Size([1, 3, 288, 456])
        # divisor = torch.nn.functional.fold(torch.nn.functional.unfold(y_ones, padsize*scale-shave*scale, stride=int(shave/2*scale)),((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), padsize*scale-shave*scale, stride=int(shave/2*scale))#torch.Size([1, 3, 288, 456])

        # y_inter = y_inter/divisor #torch.Size([1, 3, 288, 456])

        # y[...,int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)] = y_inter

        # y = torch.cat([y[...,:y.size(2)-int((padsize-h_cut)/2*scale),:],y_h_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]],dim=2) #torch.Size([1, 3, 321, 480])
        # y_w_cat = torch.cat([y_w_cut[...,:y_w_cut.size(2)-int((padsize-h_cut)/2*scale),:],y_hw_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]],dim=2) #torch.Size([1, 3, 321, 48])
        # y = torch.cat([y[...,:,:y.size(3)-int((padsize-w_cut)/2*scale)],y_w_cat[...,:,int((padsize-w_cut)/2*scale+0.5):]],dim=3) #torch.Size([1, 3, 321, 481])
        return output, y_hw_cut  # y.cuda()

    def cut_h(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):

        x_h_cut_unfold = torch.nn.functional.unfold(x_h_cut, padsize, stride=int(shave / 2)).transpose(0,
                                                                                                       2).contiguous()

        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
        y_h_cut_unfold = []
        x_h_cut_unfold.cuda()
        for i in range(x_range):
            y_h_cut_unfold.append(self.model.forward(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu())
            # y_h_cut_unfold.append(P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(1)).cpu())
        y_h_cut_unfold = torch.cat(y_h_cut_unfold, dim=0)

        y_h_cut = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize * scale, (w - w_cut) * scale), padsize * scale, stride=int(shave / 2 * scale))
        y_h_cut_unfold = y_h_cut_unfold[..., :,
                         int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
        y_h_cut_inter = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize * scale, (w - w_cut - shave) * scale), (padsize * scale, padsize * scale - shave * scale),
            stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize * scale, padsize * scale - shave * scale),
                                       stride=int(shave / 2 * scale)), (padsize * scale, (w - w_cut - shave) * scale),
            (padsize * scale, padsize * scale - shave * scale), stride=int(shave / 2 * scale))
        y_h_cut_inter = y_h_cut_inter / divisor

        y_h_cut[..., :, int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_h_cut_inter
        return y_h_cut

    def cut_w(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):

        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave / 2)).transpose(0,
                                                                                                       2).contiguous()

        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
        y_w_cut_unfold = []
        x_w_cut_unfold.cuda()
        for i in range(x_range):
            y_w_cut_unfold.append(self.model.forward(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu())
            # y_w_cut_unfold.append(P.data_parallel(self.model, x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        y_w_cut_unfold = torch.cat(y_w_cut_unfold, dim=0)

        y_w_cut = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut) * scale, padsize * scale), padsize * scale, stride=int(shave / 2 * scale))
        y_w_cut_unfold = y_w_cut_unfold[..., int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                         :].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut - shave) * scale, padsize * scale), (padsize * scale - shave * scale, padsize * scale),
            stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize * scale - shave * scale, padsize * scale),
                                       stride=int(shave / 2 * scale)), ((h - h_cut - shave) * scale, padsize * scale),
            (padsize * scale - shave * scale, padsize * scale), stride=int(shave / 2 * scale))
        y_w_cut_inter = y_w_cut_inter / divisor

        y_w_cut[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale), :] = y_w_cut_inter
        return y_w_cut

    def forward(self, x):
        return self.forward_chop(x)


class ipt(nn.Module):
    def __init__(self, n_feats=64, shift_mean=True, precision='single',
                 rgb_range=255, scale=[1], n_colors=3, patch_size=48, patch_dim=3,
                 num_heads=12, num_layers=12, num_queries=1, dropout_rate=0,
                 no_mlp=False, pos_every=False, no_pos=False, no_norm=False, conv=common.default_conv,
                 crop_batch_size=64):
        super(ipt, self).__init__()
        self.scale_idx = 0
        self.crop_batch_size = crop_batch_size
        self.patch_size = patch_size

        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(n_colors, n_feats, kernel_size),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in scale
        ])

        self.body = VisionTransformer(img_dim=patch_size,
                                      patch_dim=patch_dim,
                                      num_channels=n_feats,
                                      embedding_dim=n_feats * patch_dim * patch_dim,
                                      num_heads=num_heads, num_layers=num_layers,
                                      hidden_dim=n_feats * patch_dim * patch_dim * 4,
                                      num_queries=num_queries, dropout_rate=dropout_rate,
                                      mlp=no_mlp, pos_every=pos_every,
                                      no_pos=no_pos, no_norm=no_norm)

        self.tail = nn.ModuleList([
            nn.Sequential(
                common.Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, n_colors, kernel_size)
            ) for s in scale
        ])

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head[self.scale_idx](x)

        res = self.body(x, self.scale_idx)
        res += x

        x = self.tail[self.scale_idx](res)
        x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            num_queries,
            positional_encoding_type="learned",
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos

        if self.mlp == False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )

            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

    def forward(self, x, query_idx, con=False):

        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0,
                                                                                                           1).contiguous()

        if self.mlp == False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x

            query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
        else:
            query_embed = None

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            x = self.decoder(x, x, query_pos=query_embed)
        else:
            x = self.encoder(x + pos)
            x = self.decoder(x, x, query_pos=query_embed)

        if self.mlp == False:
            x = self.mlp_head(x) + x

        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)

        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                         stride=self.patch_dim)
            return x, con_x

        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                     stride=self.patch_dim)

        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
