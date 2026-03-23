## AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation
## Yuning Cui, Syed Waqas Zamir, Salman Khan, Alois Knoll, Mubarak Shah, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2403.14614


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from models.V6.network import ResNet34Mvt,CrossAttention,SubpixelConvLayer
from wrappers.FBPtools import FBPTools

class D3DIR_V6(FBPTools):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 decoder=True,
                 ):

        super(D3DIR_V6, self).__init__()

        self.patch_embed1 = ResNet34Mvt(inp_channels, out_channels)
        self.patch_embed2 = ResNet34Mvt(inp_channels, out_channels)
        self.patch_embed3 = ResNet34Mvt(inp_channels, out_channels)
        self.cross_attention = CrossAttention(
                     dim=1, num_heads=1, 
                     qkv_bias=False,
                     qk_scale=None, 
                     attn_drop=0., 
                     proj_drop=0.)
        
        self.cons = nn.Conv2d(3, out_channels=1, kernel_size=3, stride=2, padding=1, bias=bias)
        self.SubpixelConv = nn.Sequential(SubpixelConvLayer(in_channels=1,out_channels=1,scale_factor=2),nn.ReLU())
        self.output = nn.Conv2d(1, out_channels=3, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, noise_emb=None):
        #print('inp_img:',inp_img.shape)
        inp_img = inp_img[:,0:1,:,:]
        inp_enc_level1 = self.cons(self.patch_embed1(inp_img))
        inp_enc_level2 = self.cons(self.patch_embed2(inp_img))
        inp_enc_level3 = self.cons(self.patch_embed3(inp_img))

        out = self.cross_attention(inp_enc_level1,inp_enc_level2,inp_enc_level3)
        out = self.SubpixelConv(out)
        #print(out.shape)
        out_dec_level1 = self.output(out) + inp_img

        return torch.cat([out_dec_level1,out_dec_level1,out_dec_level1], dim=1)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# from torchvision import transforms
# from PIL import Image
# def get_data():
#     # 生成假数据
#     batch_size = 1  # 批次大小减少到 1
#     inp_channels = 1 # 输入通道数，保持为 1
#     height = 256  # 图像高度减小到 16
#     width = 256  # 图像宽度减小到 16
#     dim = 1  # 特征维度减小到 16
#     # 定义图像预处理操作
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),      # 将图像缩放为 256x256
#         transforms.ToTensor(),              # 将图像转换为 PyTorch 张量
#     ])

#     # 创建假数据 (假设 `inp_img` 是输入图像)
    
#     # 创建噪声嵌入（`noise_emb`），根据新的维度调整
#     noise_emb = torch.randn(batch_size, dim * 2 ** 2)  # 调整噪声嵌入的维度

#     # 读取图片
#     img_path = 'E:/HSI/WSY/125_gt.png'  # 替换为你的图片路径
#     img = Image.open(img_path).convert('RGB')  # 打开图片并确保是 RGB 格式
#     img =  img.convert('L') 

#     # 预处理图片
#     inp_img = transform(img)
#     img.show()

#     # 扩展维度
#     batch_size = 1
#     inp_img_batch = inp_img.unsqueeze(0).repeat(batch_size, 1, 1, 1)


#     # 查看生成的假数据
#     print("Input Image Shape:", inp_img.shape)  # 输出假数据的形状
#     print("Noise Embedding Shape:", noise_emb.shape)  # 输出噪声嵌入的形状
#     return inp_img_batch,noise_emb

# def main():

#     inp_img,noise_emb = get_data()

#     # 假设 D3DIR 类已经定义并且可以实例化
#     model = D3DIR(inp_channels=1, out_channels=1, dim=12)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params}")

#     # 使用假数据进行前向传递
#     output = model(inp_img)

#     # 输出形状
#     # print("Output Shape:", output.shape)

#     # 假设输出是一个张量，形状为 (batch_size, channels, height, width)
#     # 我们从 batch 中取出一张图像（假设 batch_size=1），并将其转换为 PIL 图像
#     output_img = output.squeeze(0)  # 去掉 batch 维度，假设输出是 [1, C, H, W] 的张量
#     output_img = output_img.detach().cpu()  # 确保数据在 CPU 上并且不参与梯度计算

#     # 转换为 PIL 图像
#     to_pil = transforms.ToPILImage()
#     output_pil = to_pil(output_img)

#     # 显示图像
#     output_pil.show()
#     #output_pil.save("E:/HSI/WSY/output/test3.png")  # 保存为 PNG 格式


#     return


# if __name__=="__main__":
    
#     main()



