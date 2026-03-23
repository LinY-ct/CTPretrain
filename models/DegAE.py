from wrappers.FBPtools import FBPTools
import torch
import models.Restormer_Backbone_arch as Restormer_Backbone_arch
import models.RCAN_Pretrain_Head_arch as RCAN_Pretrain_Head_arch


class DegAE(FBPTools):
    def __init__(self):
        super().__init__()

        net_dict = dict(inp_channels=3, out_channels=64, dim=48, num_blocks=[4, 6, 6, 8],
                        num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                        bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)

        self.netEncoder = Restormer_Backbone_arch.Restormer_Backbone(**net_dict).to('cuda')
        self.netDecoder1 = RCAN_Pretrain_Head_arch.One_Conv_Head(in_c=64,
                                                                 out_c=3, scale=1,
                                                                 require_modulation=False).to('cuda')
        # if opt.trainer_mode == 'train':
        #     self.netEncoder.train()
        #     self.netDecoder1.train()

    def forward(self, input):
        encoded_feature = self.netEncoder(input)
        fake_H = self.netDecoder1(encoded_feature)
        return fake_H

    def tune(self):
        load_path_G = "/mnt/nas/wsy/Codes/DegAE_DegradationAutoencoder/experiments/PretrainEncoder.pth"
        if load_path_G is not None:
            print('Loading model for Encoder [{:s}] ...'.format(load_path_G))
            load_net = torch.load(load_path_G)
            self.netEncoder.load_state_dict(load_net, strict=True)



