import torch
import torch.nn as nn
from wrappers.FBPtools import FBPTools
import torch.nn.functional as F
from models.hubconf import dinov2_vitb14

class dinov2(FBPTools):
    def __init__(self, embed_dim=768, decoder_embed_dim=512,  num_patches=256,mlp_ratio=4.,decoder_depth=8,
                 decoder_num_heads=16, norm_layer=nn.LayerNorm,):
        super(dinov2, self).__init__()
        self.dinov2_model = dinov2_vitb14(pretrained=True)
        # self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',pretrained=True)
            #torch.hub.load('/home/wsy/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitb14', source='local',pretrained=True))

            #torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14'))
        # for param in self.dinov2_model.parameters():
        #     param.requires_grad = False
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        #
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        #
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, decoder_embed_dim),
        #                                       requires_grad=False)  # fixed sin-cos embedding
        #
        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        #
        # self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, 14 ** 2 * 1, bias=True)  # decoder to patch

    def unpatchify(self, x):

        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 14
        h = w = 16
        # print(x.shape)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs
    # def forward_decoder(self, x):
    #     x = self.decoder_embed(x)
    #     x = x + self.decoder_pos_embed
    #     for blk in self.decoder_blocks:
    #         x = blk(x)
    #     x = self.decoder_norm(x)
    #     x = self.decoder_pred(x)
    #     # remove cls token
    #     x = x[:, 1:, :]
    #     return x
    def forward(self, imgs):
        imgs = F.interpolate(imgs, (224, 224), mode='bilinear')
        dino_output = self.dinov2_model.forward_features(imgs)
        x = dino_output['x_norm_patchtokens']
        # cls_token = dino_output['x_norm_clstoken'].expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_token, x), dim=1)
        # print(type(x))
        # print(x.shape)

        # x = self.forward_decoder(x)
        # pred = self.unpatchify(x)
        # pred = F.interpolate(pred, (256, 256), mode='bilinear')
        return x