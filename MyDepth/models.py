#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys
sys.path.append((project_directory/'omnidata/omnidata_tools/torch').as_posix())
sys.path.append((project_directory/'ZoeDepth').as_posix())
#%%
from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import MixedNYUKITTI
from modules.midas.dpt_depth import DPTDepthModel

import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch
import numpy as np
from pprint import pprint
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from read import CustomDataset
from modules.midas.dpt_depth import DPTDepthModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import numpy as np
#%%
def get_omni(pretrained_weights_path):
    checkpoint = torch.load(pretrained_weights_path,
                            # map_location=map_location
                            map_location=torch.device('cpu')
                            )
    if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
    model.load_state_dict(state_dict, strict=False)
    return model
import peft
# print(f"PEFT version: {peft.__version__}")  # debug用
from peft import LoraConfig, get_peft_model
# class WrapperZoeFormatOutput(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#     def forward(self, x):
#         y = model(x)
#         return dict(
            
#         )

from transformers import AutoImageProcessor, DPTForDepthEstimation

class ThreeDPT(nn.Module):
    def __init__(self, pretrained_weights_path):
        super().__init__()
        self.relative = get_peft_model(get_omni(pretrained_weights_path), 
                         LoraConfig(
                            r=16,  # Lora矩阵的中间维度。=r 越小，可训练的参数越少，压缩程度越高
                            lora_alpha=16,  #  LoRA 矩阵的稀疏性=非零元素的比例。lora_alpha 越小，可训练的参数越少，稀疏程度越高.
                            target_modules=['qkv'],  # 这里指定想要被 Lora 微调的模块
                            lora_dropout=0.5, # 防止过拟合，提高泛化能力
                            bias="none",  # bias是否冻结
                            )              
                        )
        
        # 正则化的方向：趋于0；要让不同地方scale差不多；矩阵要厉害一点, 趋向于0
        # self.scale = get_omni(pretrained_weights_path)
        self.scale = DPTDepthModel(backbone='vitb_rn50_384') # 与相对深度无关
        # self.scale = DPTDepthModel(backbone='vit_base_patch14_dinov2.lvd142m') # 与相对深度无关
        # self.scale = DPTDepthModel(backbone='vit_so400m_patch14_siglip_384') # 与相对深度无关
        # self.scale = get_peft_model(self.scale, 
        #                  LoraConfig(
        #                     r=64,  
        #                     lora_alpha=64,  
        #                     target_modules=['qkv'],  
        #                     lora_dropout=0.05,
        #                     bias='all', 
        #                     )              
        #                 )
        # 正则化的方向：趋于0；要让不同地方scale差不多；矩阵要厉害一点。shift是一个补充的东西，本来应该是0附近。
        self.shift = get_omni(pretrained_weights_path)
        # self.shift = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-base-nyu")
        # self.shift = DPTDepthModel(backbone='vitb_rn50_384')
        # self.shift = get_peft_model(self.shift, 
        #                  LoraConfig(
        #                     r=64,  
        #                     lora_alpha=64,  
        #                     target_modules=['qkv'],  
        #                     lora_dropout=0.05,
        #                     bias="all",  
        #                     )              
        #                 )
        # # 正则化的方向：趋于0；要让不同地方scale差不多；矩阵要厉害一点。shift是一个补充的东西，本来应该是0附近。
        # self.shift2 = get_omni(pretrained_weights_path)
        # self.shift2 = get_peft_model(self.shift, 
        #                  LoraConfig(
        #                     r=64,  
        #                     lora_alpha=64,  
        #                     target_modules=['qkv'],  
        #                     lora_dropout=0.05,
        #                     bias="all",  
        #                     )              
        #                 )
        self.relative.print_trainable_parameters()
        # self.scale.print_trainable_parameters()
        # self.shift.print_trainable_parameters()
    
    
    # 注意这个版本是 “”
    def forward(self, x):
        # print(self.relative(x))
        # _, relative = self.relative(x) 原版模型没有返回特征图出来
        relative = self.relative(x)
        scale = self.scale(x)
        shift = self.shift(x)
        from scipy import stats
        
        # print(stats.describe(relative.detach().cpu().squeeze().reshape(-1).numpy()))
        # print(stats.describe(scale.detach().cpu().squeeze().reshape(-1).numpy()))
        # print(stats.describe(shift.detach().cpu().squeeze().reshape(-1).numpy()))

        relative = relative.clamp(0, 1) # 相对深度被期望输出0-1之间的值
        # shift2 = self.shift2(x)
        # output = relative*(scale*1000/4)+shift*1000/4 # 勉强可以训练
        # output = relative*scale.mean()+shift.mean()
        
        scale = scale.clamp(0, 1)
        shift = (shift.clamp(0, 1)-0.5)*2 # 
        # scale=1000*scale
        # shift=1000*shift
        # scale = ((scale-scale.min() )/(scale.max()-scale.min()).clamp(min=1e-6)).clamp(min=1e-6)
        # shift = ((shift-shift.min() )/(shift.max()-shift.min()).clamp(min=1e-6)).clamp(min=1e-6)
        # shift = (shift-shift.mean() )/(shift.std().clamp(min=1e-6))
        
        
        # shift = (shift-0.5)*2
        output = (relative)*scale*20+shift*0.1 # 先验: 20m是因为赛方提供最远距离是20; 0.1米是大家模型能达到的误差
        # output = (relative)*scale*20*8+shift*0.1*24 # 先验: 20m是因为赛方提供最远距离是20; 0.1米是大家模型能达到的误差
        # output = (relative+shift2*0.2).clamp(0, 1)*scale.mean()*20+shift*0.2 # 先验: 20m是因为赛方提供最远距离是20; 0.1米是大家模型能达到的误差
        # return output.squeeze(dim=1) # 原本维度为 b, 1, w, h
        return dict(
            # metric_depth=output.squeeze(dim=1)
            # metric_depth=output.unsqueeze(dim=1),
            metric_depth=output,
            rel_depth=relative
        )
    
    
    
    # from peft import LoraConfig, get_peft_model
    # config = LoraConfig(
    #     r=16,  # Lora矩阵的中间维度。 决定了有多少可训练参数
    #     lora_alpha=16,  # ？。也决定了有多少可训练参数
    #     target_modules=['qkv'],  # 这里指定想要被 Lora 微调的模块
    #     lora_dropout=0.1,
    #     bias="none",  # bias是否冻结
    #     # modules_to_save=["classifier"], 这里指定不想要被 Lora 微调，但是也不想要冻结，想要全量微调的模块
    # )
    # model.core.core = get_peft_model(model.core.core, config)
    # model.core.core.print_trainable_parameters()  # 检查PEFT使用后模型要被我们训练的参数量
# ThreeDPT('/home/ycm/repos/competitons/depth/MyDepth/omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt')    
    
#%%
def get_zoe_single_head_with_omni(pretrained_weights_path):
    config = get_config(
    "zoedepth",  # model_name  允许的choice ['zoedepth', 'zoedepth_nk']
    # "zoedepth_nk",  # model_name  允许的choice ['zoedepth', 'zoedepth_nk']
                    "train",     # mode 
                       "mix",  # dataset
                    )
    if config.use_shared_dict:
        shared_dict = mp.Manager().dict()
    else:
        shared_dict = None
    config.shared_dict = shared_dict

    config.batch_size = config.bs
    config.mode = 'train'
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)
    config
    
    config.midas_model_type = "DPT_Hybrid"
    model = build_model(config)
    model = model.to('cuda')
    
    # 尝试把头也加进去？
    # state_dict = 
    # res = model.load_state_dict(state_dict, strict=False)
    # print(res)
    
    # map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')

    # model.core.core = get_omni(pretrained_weights_path)
    checkpoint = torch.load(pretrained_weights_path,
                            # map_location=map_location
                            map_location=torch.device('cpu')
                            )
    model.core.core.load_state_dict(checkpoint, strict=False)
    
    import peft
    print(f"PEFT version: {peft.__version__}")  # debug用
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=16,  # Lora矩阵的中间维度。 决定了有多少可训练参数
        lora_alpha=16,  # ？。也决定了有多少可训练参数
        target_modules=['qkv'],  # 这里指定想要被 Lora 微调的模块
        lora_dropout=0.1,
        bias="none",  # bias是否冻结
        # modules_to_save=["classifier"], 这里指定不想要被 Lora 微调，但是也不想要冻结，想要全量微调的模块
    )
    model.core.core = get_peft_model(model.core.core, config)
    model.core.core.print_trainable_parameters()  # 检查PEFT使用后模型要被我们训练的参数量
    
    
    # set_require_grad(model.core.core, False) # 冻结
    # set_require_grad(model.core.core, True) # 冻结
    return model

def set_require_grad(model, require_grad=False):
    for param in model.parameters():
        param.requires_grad = require_grad

#%%
# 定义网络

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.name = 'MyNetwork'
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(32 * 12 * 12, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MyNetwork_large(nn.Module):
    def __init__(self):
        super(MyNetwork_large, self).__init__()
        self.name = 'MyNetwork_large'
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(256 * 12 * 12, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MyNetwork_large_bn(nn.Module):
    def __init__(self):
        super(MyNetwork_large_bn, self).__init__()
        self.name = 'MyNetwork_large_bn'
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)  
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)  
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(256 * 12 * 12, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MyNetwork_large_384(nn.Module):
    def __init__(self):
        super(MyNetwork_large_384, self).__init__()
        self.name = 'MyNetwork_large_384'
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2)
        x = nn.functional.interpolate(x, size=[384, 384], mode='nearest')
        return x
    
import timm

#第一种改法    
class ResNet_v1(nn.Module):
    def __init__(self):
        super(ResNet_v1, self).__init__()
        self.name = 'ResNet18_v1'
        self.conv = nn.Conv2d(256, 3, kernel_size=1)
        self.resnet = timm.create_model('resnet18', pretrained=True)
        # self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # 修改最后一层全连接层以适应输出为2维
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        return x

#第二种改法   
class ResNet_v2(nn.Module):
    def __init__(self):
        super(ResNet_v2, self).__init__()
        self.name = 'ResNet18_v2'
        self.resnet = timm.create_model('resnet18', pretrained=True)
        # Modify the first convolutional layer to accept [8, 256, 12, 12] input
        self.resnet.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the last fully connected layer to output 2 features
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        self.name = 'U_Net'
        self.conv0 = nn.Conv2d(256, 3, kernel_size=1, stride=1)
        self.u_net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
        self.conv1 = nn.Conv2d(1, 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.u_net(x)
        x = self.conv1(x)
        return x
from modules.midas.blocks import forward_vit
class OmniScale(nn.Module):
    def __init__(self, pretrained_weights_path, head=MyNetwork_large()):
        super().__init__()
        self.omni = get_omni(pretrained_weights_path)
        set_require_grad(self.omni, False)
        self.head = head
        set_require_grad(self.head, True)
        
    def forward(self, x):
        if self.omni.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.omni.pretrained, x)

        layer_1_rn = self.omni.scratch.layer1_rn(layer_1)
        layer_2_rn = self.omni.scratch.layer2_rn(layer_2)
        layer_3_rn = self.omni.scratch.layer3_rn(layer_3)
        layer_4_rn = self.omni.scratch.layer4_rn(layer_4)

        path_4 = self.omni.scratch.refinenet4(layer_4_rn)
        path_3 = self.omni.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.omni.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.omni.scratch.refinenet1(path_2, layer_1_rn)

        relative_depth_map = self.omni.scratch.output_conv(path_1)
        relative_depth_map = relative_depth_map.squeeze(dim=1)
        relative_depth_map = relative_depth_map.clamp(0, 1)
        
        result = self.head(layer_4_rn)
        scale, shift = result[:, 0], result[:, 1]
        scale = scale.unsqueeze(1).unsqueeze(2)  # 扩展 scale 的维度
        shift = shift.unsqueeze(1).unsqueeze(2)  # 扩展 shift 的维度
        absolute_depth_map = relative_depth_map * scale + shift  # relative_depth_map的size为torch.Size([4, 512, 512])
 

        return dict(
            # metric_depth=output.squeeze(dim=1)
            metric_depth=absolute_depth_map,
            rel_depth=relative_depth_map
        )


class WeightEnsemble(nn.Module):
    def __init__(self, models):
        super(WeightEnsemble, self).__init__()

        # Freeze the parameters of the input models
        for model in models:
            set_requires_grad(model, False)

        # Initialize weights for linear combination
        self.weights = nn.Parameter(torch.ones(len(models))/len(models))

        # Store the input models
        # self.models = nn.ModuleList(models)
        self.models = models # 不需要pytorch保存

    def forward(self, *inputs):
        # Disable gradient computation for the input models
        with torch.no_grad():
            model_outputs = [model(*inputs) for model in self.models]
        # Linear combination using the weights
        weighted_sum = sum(w * output for w, output in zip(self.weights, model_outputs))
        return weighted_sum
    
        # 模型多的时候用
        # # Stack the outputs along a new dimension
        # stacked_outputs = torch.stack(model_outputs, dim=-1)
        # # Apply weights using matrix multiplication
        # weighted_sum = torch.matmul(stacked_outputs, self.weights)
        # return weighted_sum.squeeze(dim=-1) 
    

        