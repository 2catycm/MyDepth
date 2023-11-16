#%%
# 定义基本目录+import目录
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys
sys.path.append((project_directory/'omnidata/omnidata_tools/torch').as_posix())
sys.path.append((project_directory/'ZoeDepth').as_posix())
sys.path.append((project_directory).as_posix())
#%%
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
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
exp_id = "复现实验"
model_name = "ZoeDepth_Omni"
running_path = this_directory/f"./runs/{exp_id}"  # 运行时保存的位置
running_path.mkdir(parents=True, exist_ok=True)
save_head_to = lambda epoch:(running_path/f"{model_name}_{epoch}.pth").as_posix()  # head保存的位置

pretrained_weights_path = project_directory/"omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt"

system_data_path = Path("/data/projects/depth").resolve()
dataset_directory = system_data_path/"5.跨场景单目深度估计/训练集/replica_fullplus/"
dataset_path_rgb, dataset_path_depth = dataset_directory/'rgb', dataset_directory/'depth_zbuffer'

pretrained_weights_path, dataset_path_rgb, dataset_path_depth = pretrained_weights_path.as_posix(), dataset_path_rgb.as_posix(), dataset_path_depth.as_posix()
pretrained_weights_path, dataset_path_rgb, dataset_path_depth
#%%
# 定义训练参数，方便调参
lr = 3e-4
# batch_size = 1024
# batch_size = 128
# batch_size = 4 # 5G 显存
# batch_size = int(4 * (80*4/5) * 0.97) # 
# single_gpu_memory = 80
single_gpu_memory = 24
batch_size = int(4 * (single_gpu_memory*1/5) * 0.97*1.1) # 
# batch_size = int(4 * (80*1/5) * 0.97*1.1/2) # 
print(f"batch_size: {batch_size}")
# save_epoch = 4
save_steps = 30
num_epochs = 1

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
# from
# criterion = nn.L1Loss()
# criterion = nn.MSELoss()
from losses import ValidatedLoss,  CompetitionLoss
criterion = ValidatedLoss(basic_loss=CompetitionLoss(), lower=0.1, upper=20)
criterion = criterion.to(device)
# criterion = nn.DataParallel(criterion)
import sam.sam as sam
# optimizer = optim.AdamW(model.parameters(), lr=lr)
base_optimizer = torch.optim.AdamW
# optimizer = sam.SAM(model.parameters(), base_optimizer, lr=0.001, momentum=0.9)
optimizer = sam.SAM(model.parameters(), base_optimizer, 
                    lr=lr)


#%%
# 训练网络
# def post_process(output):
    

import tqdm
bar = tqdm.tqdm(range(num_epochs), colour='green', leave=False, position=0)
for epoch in bar:
    inner_bar = tqdm.tqdm(train_data_loader, colour='yellow', leave=False, position=1)
    epoch_loss_sum = 0
    i_log = 0
    for images, depths_gt in inner_bar:
        # 似乎是to之后会导致device问题
        images = images.to(device)
        depths_gt = depths_gt.to(device)
        # debug
        # print(images.shape, depths_gt.shape)

        b, c, h, w = images.size()
        output = model(images)
        
        pred_depths = output['metric_depth']

        
        # SAM
        # first forward-backward pass
        loss = criterion(depths_gt, pred_depths)  # use this loss for any training statistics
        loss.backward()
        optimizer.first_step(zero_grad=True)
        # grad_norm = np.array([p.norm().item()
        #     for p in head.parameters()
        # ]).mean()
        inner_bar.set_postfix(loss=loss.item(), 
                            #   grad_norm=grad_norm
                              )
        epoch_loss_sum+=loss.item()
        bar.set_postfix(Epoch=epoch, loss=epoch_loss_sum/(i_log+1))
        
        # second forward-backward pass
        criterion(depths_gt, model(images)['metric_depth']).backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)
        
        if i_log%save_steps == 0:
            torch.save(model.state_dict(), save_head_to(epoch*len(train_data_loader)+i_log))
        i_log+=1

    
    # if epoch%save_steps == 0:
    #     torch.save(model.state_dict(), save_head_to(epoch))

# %%
