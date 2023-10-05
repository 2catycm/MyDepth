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
pretrained_weights_path = project_directory/"omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt"
model = models.get_zoe_single_head_with_omni(pretrained_weights_path)

state_dict = torch.load('./runs/3/ZoeDepth_Omni_714.pth')
model.load_state_dict(state_dict, strict=False)

model_method = lambda x:model(x)['metric_depth']
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
                        image_size=[384, 512])
# testloader = DataLoader(dataset, batch_size=int(32*4))
testloader = DataLoader(dataset, batch_size=int(32*2))

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
