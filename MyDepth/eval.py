#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent

import sys
sys.path.append((project_directory/'ZoeDepth').as_posix())
sys.path.append((project_directory).as_posix())
sys.path.append((project_directory/"MyDepth").as_posix())

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#%%
# 加载模型为model
import models
from modules.midas.dpt_depth import DPTDepthModel
# 路径

pretrained_weights_path = project_directory/"omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt"

# pretrained_head = project_directory/"MyDepth/runs/1/SimpleMetricHead_60.pth"
# pretrained_head = project_directory/"MyDepth/runs/2/SimpleMetricHead_2826.pth"

# 找到最新写入的 checkpoint ls -t | head -1
pretrained_head = project_directory/"MyDepth/runs/2/SimpleMetricHead_1024.pth"
# omni
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
# model = DPTDepthModel(backbone='vitl16_384') # DPT Large
omni = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint
omni.load_state_dict(state_dict)
omni = omni.to(device)
# head
head = models.MyNetwork()
head = head.to(device)
if pretrained_head is not None:
    checkpoint_head = torch.load(pretrained_head, map_location=map_location)
    head.load_state_dict(checkpoint_head,strict=True)

# parallelize
# omni = nn.DataParallel(omni)
# head = nn.DataParallel(head)
    
def model_method(x):
    # print(x.shape)
    feature_map, output = omni(x) #
    output = output.squeeze(dim=1)
    # output = output.clamp(min=0, max=1) #限制0到1之间
    # output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0) #双线性插值放大到512x512
    output = output.clamp(0, 1)
    relative_depth_map = output
    
    result = head(feature_map)           #result的size为4x2，4是batch_size
    scale, shift = result[:, 0], result[:, 1]
    scale = scale.unsqueeze(1).unsqueeze(2)  # 扩展 scale 的维度
    shift = shift.unsqueeze(1).unsqueeze(2)  # 扩展 shift 的维度
    #print(relative_depth_map.shape, scale.shape)
    absolute_depth_map = relative_depth_map * scale + shift
    

    # return absolute_depth_map*1000 # 比赛要求
    return absolute_depth_map # 比赛要求


#%%
import os
from PIL import Image
from torchvision import transforms
import cv2
from torch.utils.data import Dataset
import torch
from read import CustomDataset
from torch.utils.data import DataLoader
from losses import CompetitionLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#加载数据集
dataset_directory = project_directory/"data/szh/5.跨场景单目深度估计/训练集/taskonomy"
dataset_path_rgb, dataset_path_depth = dataset_directory/'rgbs', dataset_directory/'depths'
dataset_path_rgb, dataset_path_depth = dataset_path_rgb.as_posix(), dataset_path_depth.as_posix()

dataset = CustomDataset(dataset_path_rgb, dataset_path_depth, 
                        image_size=384)
testloader = DataLoader(dataset, batch_size=32)

#计算指标
# loss_fn = CompetitionLoss()
# loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()
loss = 0
from tqdm import tqdm
bar = tqdm(enumerate(testloader), desc="eval", leave=False, position=0)
with torch.no_grad():
    for i, (inputs, ground_truth) in bar:
        inputs = inputs.to(device)
        ground_truth = ground_truth.to(device)
        result = model_method(inputs)
        loss += loss_fn(result, ground_truth)
        bar.set_postfix({'loss': loss/(i+1)})
print(loss//len(testloader))
