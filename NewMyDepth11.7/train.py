# 定义训练参数
lr = 1e-5
batch_size = 8
save_steps = 100
num_epochs = 1
log_steps = 100
cuda_num = 1
experiment_name = f'resnet'
seed = 1

cuda_name = 'cuda:' + str(cuda_num)
pretrained_weights_path = 'pretrained_models/omnidata_dpt_depth_v2.ckpt'
 
# 选择网络结构
from models import *
head = ResNet_v1()

# # 数据集replica_fullplus
# save_path = 'logs/' + 'saves_replica_fullplus/' + experiment_name + '/'
# dataset_path_rgb = 'replica_fullplus/rgb/replica'
# dataset_path_depth = 'replica_fullplus/depth_zbuffer/replica'

# 数据集taskonomy
save_path = 'logs/' + 'saves_taskonomy/' + experiment_name + '/'
dataset_path_rgb = 'taskonomy/rgbs'
dataset_path_depth = 'taskonomy/depths'

print('-------------------------------------------')
print(experiment_name)
print(cuda_name)
print(dataset_path_rgb)
print(head.name)
print('batch_size:', batch_size)
print('')

import torch
import os
import torch.optim as optim
from read import CustomDataset
from modules.midas.dpt_depth import DPTDepthModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn.functional as F

# 输入图片，利用预训练模型得到feature_map和相对深度图
def get(x, model):
    feature_map, output = model(x)
    output = output.squeeze(dim=1)
    output = output.clamp(0, 1)
    return feature_map, output

#固定随机数种子
def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

seed_torch(seed)
warnings.filterwarnings("ignore")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 加载数据集
dataset = CustomDataset(dataset_path_rgb, dataset_path_depth, image_size=384)
train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 加载预训练模型
map_location = (lambda storage, loc: storage.cuda(cuda_num)) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model = model.to(device)

# 冻结model中所有参数
model.eval()
for param in model.parameters():
    param.requires_grad = False

# 创建网络 优化器传head参数
head = head.to(device)
optimizer = optim.AdamW(head.parameters(), lr=lr)

#tensorboard将训练loss可视化
writer = SummaryWriter('runs', filename_suffix='_' + experiment_name)

epoch_num = 0
for epoch in range(num_epochs):
    epoch_num += 1
    print('epoch_num:', epoch_num)

    epoch_loss_sum = 0
    loss_sum = 0
    inner_bar = tqdm(train_data_loader, leave=False, position=1, ncols = 150)
    i_log = 0
    for inputs, ground_truth in inner_bar:

        inputs = inputs.to(device)
        ground_truth = ground_truth.to(device)

        # 得到feature_map和相对深度图
        feature_map, relative_depth_map = get(inputs, model)

        # 计算绝对深度图
        result = head(feature_map)  # result的size为4x2，4是batch_size
        
        # 输出两个值
        scale, shift = result[:, 0], result[:, 1]
        scale = scale.unsqueeze(1).unsqueeze(2)  # 扩展 scale 的维度
        shift = shift.unsqueeze(1).unsqueeze(2)  # 扩展 shift 的维度
        absolute_depth_map = relative_depth_map * scale + shift  # relative_depth_map的size为torch.Size([4, 512, 512])
        
        # #每个点都一个scale和shift
        # scale, shift = result[:, 0, :, :], result[:, 1, :, :]
        # absolute_depth_map = relative_depth_map * scale + shift  # relative_depth_map的size为torch.Size([4, 512, 512])

        # 保存中间图片
        if i_log % log_steps == 0:
            plt.imsave(save_path + f'{epoch_num}_{i_log}_relative_depth_map_test.png', relative_depth_map[0].detach().cpu().squeeze(), cmap='viridis')
            plt.imsave(save_path + f'{epoch_num}_{i_log}_absolute_depth_map_test.png', absolute_depth_map[0].detach().cpu().squeeze(), cmap='viridis')
            plt.imsave(save_path + f'{epoch_num}_{i_log}_ground_truth_test.png', ground_truth[0].detach().cpu().squeeze(), cmap='viridis')
        i_log += 1

        valid = (ground_truth > 0.1) & (ground_truth < 20)
        loss = torch.mean(torch.abs(absolute_depth_map[valid] - ground_truth[valid]) / ground_truth[valid])   
        
        inner_bar.set_postfix(loss = loss.item())
        epoch_loss_sum += loss.item()
        loss_sum += loss.item()

        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # #保存模型参数
        if i_log % save_steps == 0:
           torch.save(head.state_dict(), save_path + f'{head.name}_{epoch_num}_{i_log}_{loss_sum / save_steps}.pth')
           writer.add_scalar(experiment_name, loss_sum / save_steps, i_log)
           loss_sum = 0
    torch.save(head.state_dict(), save_path + f'{head.name}_final_{epoch_num}_{loss_sum / (i_log % save_steps)}.pth')
    writer.add_scalar(experiment_name, loss_sum / (i_log % save_steps), i_log)

writer.close()