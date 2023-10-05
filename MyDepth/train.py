#%%
# 定义基本目录+import目录
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys
sys.path.append((project_directory/'omnidata/omnidata_tools/torch').as_posix())
sys.path.append((project_directory/'ZoeDepth').as_posix())
#%%
# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from read import CustomDataset
from modules.midas.dpt_depth import DPTDepthModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
#%%
# 定义数据路径
# pretrained_weights_path = './pretrained_models/' + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
# dataset_path_rgb, dataset_path_depth = 'replica_fullplus/rgb', 'replica_fullplus/depth_zbuffer'

exp_id = 1
model_name = "ZoeDepth_Omni"
running_path = this_directory/f"./runs/{exp_id}"  # 运行时保存的位置
running_path.mkdir(parents=True, exist_ok=True)
save_head_to = lambda epoch:(running_path/f"{model_name}_{epoch}.pth").as_posix()  # head保存的位置

pretrained_weights_path = project_directory/"omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt"

dataset_directory = project_directory/"data/szh/5.跨场景单目深度估计/训练集"
dataset_path_rgb, dataset_path_depth = dataset_directory/'replica_fullplus/replica_fullplus/rgb', dataset_directory/'replica_fullplus/replica_fullplus/depth_zbuffer'

pretrained_weights_path, dataset_path_rgb, dataset_path_depth = pretrained_weights_path.as_posix(), dataset_path_rgb.as_posix(), dataset_path_depth.as_posix()
pretrained_weights_path, dataset_path_rgb, dataset_path_depth
#%%
# 定义训练参数，方便调参
lr = 3e-4
# batch_size = 1024
# batch_size = 128
# batch_size = 4 # 5G 显存
# batch_size = int(4 * (80*4/5) * 0.97) # 
batch_size = int(4 * (80*1/5) * 0.97) # 
print(f"batch_size: {batch_size}")
save_epoch = 4
num_epochs = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#%%
#加载数据集
dataset = CustomDataset(dataset_path_rgb, dataset_path_depth, image_size=[384, 512])
train_data_loader = DataLoader(dataset, batch_size=batch_size)

#%%
import models
model = models.get_zoe_single_head_with_omni(pretrained_weights_path)
model = model.to(device)
# model = nn.DataParallel(model)
# model.core.core = torch.compile(model.core.core)

#%%
criterion = nn.L1Loss()
criterion = criterion.to(device)
# criterion = nn.DataParallel(criterion)

optimizer = optim.AdamW(model.parameters(), lr=lr)

#%%
# 训练网络
# def post_process(output):
    
    
import tqdm
bar = tqdm.tqdm(range(num_epochs), colour='green', leave=False, position=0)
for epoch in bar:
    inner_bar = tqdm.tqdm(train_data_loader, colour='yellow', leave=False, position=1)
    for images, depths_gt in inner_bar:
        # 似乎是to之后会导致device问题
        images = images.to(device)
        depths_gt = depths_gt.to(device)
        # debug
        # print(images.shape, depths_gt.shape)

        b, c, h, w = images.size()
        output = model(images)
        
        pred_depths = output['metric_depth']
        
        
        # debug
        # print(images.shape, depths_gt.shape, pred_depths.shape)
        

        #计算loss
        loss = criterion(pred_depths, depths_gt)
        # print('\nloss:', loss.item())
        inner_bar.set_postfix(loss=loss.item())
        #更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    bar.set_postfix(Epoch=epoch)
    
    if epoch%save_epoch == 0:
        torch.save(model.state_dict(), save_head_to(epoch))

# %%