#%%
# 定义基本目录+import目录
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys
sys.path.append((project_directory/'omnidata/omnidata_tools/torch').as_posix())
#%%
# 定义数据路径
# pretrained_weights_path = './pretrained_models/' + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
# dataset_path_rgb, dataset_path_depth = 'replica_fullplus/rgb', 'replica_fullplus/depth_zbuffer'

exp_id = 2
model_name = "SimpleMetricHead"
running_path = this_directory/f"./runs/{exp_id}"  # 运行时保存的位置
running_path.mkdir(parents=True, exist_ok=True)
save_head_to = lambda epoch:(running_path/f"{model_name}_{epoch}.pth").as_posix()  # head保存的位置


# pretrained_head = this_directory/"saves/SimpleMetricHead.pth"
# pretrained_head = this_directory/"runs/1/SimpleMetricHead_120.pth"
pretrained_head = this_directory/"runs/2/SimpleMetricHead_2826.pth"

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
# batch_size = int(4 * (80*1/5) * 0.97*1.5) # 
batch_size = int(4 * (80*1/5) * 0.97*1.8) # 
# batch_size = int(4 * (80*1/5) * 0.97*3) # 
print(f"batch_size: {batch_size}")
save_steps = 10
num_epochs = 100

log_steps = 100

#%%
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


from models import MyNetwork

#输入图片，利用预训练模型得到feature_map和相对深度图
def get(x, model):
    #推理及处理数据
    # test(model, x)
    # x.shape = (b, 3, 512, 512)
    feature_map, output = model(x) # 
    # debug
    # print(output.shape, feature_map.shape)
    
    output = output.squeeze(dim=1)
    # output = output.clamp(min=0, max=1) #限制0到1之间
    # output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0) #双线性插值放大到512x512
    output = output.clamp(0, 1)
    # output = 1 - output           #这个不知道要不要加 距离和亮度成反比
    
    # 视差与深度的关系？
    # 数据集用的是哪个？
    # model想算的是哪个？ omnidata 预测的是视差

    return feature_map, output

#忽略警告
# warnings.filterwarnings("ignore")

#%%
#加载数据集
dataset = CustomDataset(dataset_path_rgb, dataset_path_depth, image_size=384)
train_data_loader = DataLoader(dataset, batch_size=batch_size)
#%%
len(dataset)
# dataset_path_rgb


#%%
#加载预训练模型
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = DPTDepthModel(backbone='vitl16_384') # DPT Large
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
# model = nn.DataParallel(model)

#%%
model.eval()
# model.require_grad_(False)
# 冻结 model中所有参数
for param in model.parameters():
    param.requires_grad = False
#%%
# 测试是不是所有device都行
# print(model.device_ids)
def test(model, x):
    print(x.device, model.device_ids)
    for d in range(4):
        print(d)
        a, b = model(x.to(f'cuda:{d}'))
# test(model, dataset[0][0].unsqueeze(0))
#%%
#创建网络 优化器传head参数
head = MyNetwork()
head = head.to(device)
if pretrained_head is not None:
    checkpoint_head = torch.load(pretrained_head, map_location=map_location)
    head.load_state_dict(checkpoint_head,strict=True)
# head = nn.DataParallel(head)

criterion = nn.L1Loss()
criterion = nn.L1Loss()
criterion = criterion.to(device)
# criterion = nn.DataParallel(criterion)

optimizer = optim.AdamW(head.parameters(), lr=lr)


#%%

# 训练网络
# model = torch.compile(model)
# head = torch.compile(head)
import tqdm
bar = tqdm.tqdm(range(num_epochs), colour='green', leave=False, position=0)
for epoch in bar:
    epoch_loss_sum = 0
    inner_bar = tqdm.tqdm(train_data_loader, colour='yellow', leave=False, position=1)
    i_log = 0
    for inputs, ground_truth in inner_bar:
        # 似乎是to之后会导致device问题
        inputs = inputs.to(device)
        ground_truth = ground_truth.to(device)

        #得到feature_map和相对深度图
        # with torch.no_grad():
        feature_map, relative_depth_map = get(inputs, model)

        # 计算绝对深度图
        result = head(feature_map)           #result的size为4x2，4是batch_size
        scale, shift = result[:, 0], result[:, 1]
        scale = scale.unsqueeze(1).unsqueeze(2)  # 扩展 scale 的维度
        shift = shift.unsqueeze(1).unsqueeze(2)  # 扩展 shift 的维度
        #print(relative_depth_map.shape, scale.shape)
        absolute_depth_map = relative_depth_map * scale + shift   #relative_depth_map的size为torch.Size([4, 512, 512])

        # debug 查看正不正常
        #print(absolute_depth_map.shape, ground_truth.shape)
        # print(absolute_depth_map[0])
        # print(ground_truth[0])
        from PIL import Image
        import matplotlib.pyplot as plt
        if i_log % log_steps ==0:
            plt.imsave('saves/absolute_depth_map_test.png', absolute_depth_map[0].detach().cpu().squeeze(),cmap='viridis')
            plt.imsave('saves/ground_truth_test.png', ground_truth[0].detach().cpu().squeeze(),cmap='viridis')
        i_log+=1
        # print(scale, shift)
        
        #计算loss
        loss = criterion(absolute_depth_map, ground_truth)
        # print('\nloss:', loss.item())
        grad_norm = np.array([p.norm().item()
            for p in head.parameters()
        ]).mean()
        inner_bar.set_postfix(loss=loss.item(), grad_norm=grad_norm)
        epoch_loss_sum+=loss.item()
        #更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 
        # print([p for p in head.conv1.parameters()])
        # 看看grad的norm是不是0？
        # print()
        # head 所有梯度的
        if i_log%save_steps == 0:
            torch.save(head.state_dict(), save_head_to(epoch*len(train_data_loader)+i_log))
        
        

        
    bar.set_postfix(Epoch=epoch, loss=epoch_loss_sum/len(train_data_loader))
    


# %%
