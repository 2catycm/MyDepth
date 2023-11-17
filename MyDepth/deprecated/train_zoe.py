# %%
# 定义基本目录+import目录
from pathlib import Path

this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys

sys.path.append((project_directory / "omnidata/omnidata_tools/torch").as_posix())
sys.path.append((project_directory / "ZoeDepth").as_posix())
sys.path.append((project_directory).as_posix())
# %%
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from read import CustomDataset
from modules.midas.dpt_depth import DPTDepthModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings


# 固定随机数种子
def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


seed = 114514
seed_torch(seed)

# %%
# 定义数据路径
# exp_id = "复现实验"
# exp_id = "不使用PEFT-不使用lr_schedule-仅训练两轮"
exp_id = "使用lr_schedule-训练5轮-双数据集"
# model_name = "ZoeDepth_Omni"
model_name = "ThreeDPT"
system_data_path = Path("/data/projects/depth").resolve()

# running_path = this_directory/f"./runs/{exp_id}"  # 运行时保存的位置
running_path = system_data_path / f"./runs/{exp_id}"  # 运行时保存的位置
if not running_path.is_symlink():  # 如果是软连接，也是存在
    running_path.mkdir(parents=True, exist_ok=True)

logs_path = this_directory / f"./logs/{exp_id}"
logs_path.mkdir(parents=True, exist_ok=True)

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
writer = SummaryWriter(
    logs_path/'tensorboard',
)
#    filename_suffix='_' + exp_id)


save_head_to = lambda epoch: (
    running_path / f"{model_name}_{epoch}.pth"
).as_posix()  # head保存的位置

pretrained_weights_path = (
    project_directory
    / "omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt"
)


dataset_directory = system_data_path / "5.跨场景单目深度估计/训练集/replica_fullplus/"
dataset_path_rgb1, dataset_path_depth1 = (
    dataset_directory / "rgb",
    dataset_directory / "depth_zbuffer",
)

dataset_directory = system_data_path / "5.跨场景单目深度估计/训练集/taskonomy/"
dataset_path_rgb2, dataset_path_depth2 = (
    dataset_directory / "rgbs",
    dataset_directory / "depths",
)


(
    pretrained_weights_path,
    dataset_path_rgb1,
    dataset_path_depth1,
    dataset_path_rgb2,
    dataset_path_depth2,
) = (
    pretrained_weights_path.as_posix(),
    dataset_path_rgb1.as_posix(),
    dataset_path_depth1.as_posix(),
    dataset_path_rgb2.as_posix(),
    dataset_path_depth2.as_posix(),
)
# pretrained_weights_path, dataset_path_rgb, dataset_path_depth
# %%
# 定义训练参数，方便调参
lr = 3e-6
# lr = 3e-4
# lr = 1e-3
# lr = 1e-2
# lr = 1e-1 # ZoeDepthOmni可用
# lr = 0.5
# lr = 0.9

# batch_size = 4 # 5G 显存
# single_gpu_memory = 80
single_gpu_memory = 24
n_gpus = 1
# ZoeDepthOmni
# batch_size = int(n_gpus * (single_gpu_memory*1/5) * 0.97*1.1*24564/14814) # 全量微调?
# batch_size = int(
#     n_gpus * (single_gpu_memory * 1 / 5) * 0.95 * 1.1 * 24564 / 16334 * 24564 / 14322
# )  # PEFT?
# ThreeDPT
batch_size = int(
    n_gpus * (single_gpu_memory * 1 / 5) * 0.95 * 1.1 * 24564 / 16334 * 24564 / 14322/4 *24564/17530
)  

print(f"batch_size: {batch_size}")
# save_epoch = 4
save_steps = 120 * 3
log_steps = 30
# num_epochs = 2
num_epochs = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%
# 加载数据集
from torch.utils.data import DataLoader, ConcatDataset

dataset1 = CustomDataset(dataset_path_rgb1, dataset_path_depth1, image_size=[384, 512])
dataset2 = CustomDataset(dataset_path_rgb2, dataset_path_depth2, image_size=[384, 512])
# dataset = dataset2  # taskonomy
dataset = ConcatDataset([dataset1, dataset2])
train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
import models

# model = models.get_zoe_single_head_with_omni(pretrained_weights_path)
model = models.ThreeDPT(pretrained_weights_path)
model = model.to(device)
# model = nn.DataParallel(model)
# model.core.core = torch.compile(model.core.core)

# %%
# checkpoint = torch.load("/data/projects/depth/runs/复现实验" + "/ZoeDepth_Omni_660.pth")
# model.load_state_dict(checkpoint)
# model = nn.DataParallel(model)
# %%
# from
# criterion = nn.L1Loss()
# criterion = nn.MSELoss()
from losses import ValidatedLoss, CompetitionLoss, REL

criterion = ValidatedLoss(basic_loss=CompetitionLoss(), lower=0.1, upper=20)
# criterion = ValidatedLoss(basic_loss=REL(), lower=0.1, upper=20)
criterion = criterion.to(device)
# criterion = nn.DataParallel(criterion)
import sam.sam as sam

# optimizer = optim.AdamW(model.parameters(), lr=lr)
# base_optimizer = torch.optim.AdamW
base_optimizer = torch.optim.SGD
# optimizer = sam.SAM(model.parameters(), base_optimizer, lr=0.001, momentum=0.9)
optimizer = sam.SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.9)
from torch.optim.lr_scheduler  import ExponentialLR, MultiStepLR, CosineAnnealingWarmRestarts
# scheduler1 = ExponentialLR(optimizer, gamma=0.9)
# scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
# %%
# 训练网络
# def post_process(output):


import tqdm

bar = tqdm.tqdm(range(num_epochs), colour="green", leave=False, position=0)
for epoch in bar:
    inner_bar = tqdm.tqdm(train_data_loader, colour="yellow", leave=False, position=1)
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

        pred_depths = output["metric_depth"]

        # SAM
        # first forward-backward pass
        loss = criterion(
            depths_gt, pred_depths
        )  # use this loss for any training statistics
        loss.backward()
        optimizer.first_step(zero_grad=True)
        # grad_norm = np.array([p.norm().item()
        #     for p in head.parameters()
        # ]).mean()
        inner_bar.set_postfix(
            loss=loss.item(),
            #   grad_norm=grad_norm
        )
        epoch_loss_sum += loss.item()
        bar.set_postfix(Epoch=epoch, loss=epoch_loss_sum / (i_log + 1))
        writer.add_scalar(f"loss_{exp_id}", epoch_loss_sum / (i_log + 1), epoch * len(train_data_loader) + i_log)

        # second forward-backward pass
        criterion(
            depths_gt, model(images)["metric_depth"]
        ).backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)

        if i_log % save_steps == 0:
            torch.save(
                model.state_dict(), save_head_to(epoch * len(train_data_loader) + i_log)
            )
        from PIL import Image
        import matplotlib.pyplot as plt

        if i_log % log_steps == 0:
            plt.imsave(
                (logs_path / "absolute_depth_map_test.png").as_posix(),
                pred_depths[0].detach().cpu().squeeze(),
                cmap="viridis",
            )
            plt.imsave(
                (logs_path / "ground_truth_test.png").as_posix(),
                depths_gt[0].detach().cpu().squeeze(),
                cmap="viridis",
            )
            from scipy import stats
            print(stats.describe(pred_depths[0].detach().cpu().squeeze().reshape(-1).numpy()))
            print(stats.describe(depths_gt[0].detach().cpu().squeeze().reshape(-1).numpy()))


        i_log += 1
        scheduler.step()
        writer.add_scalar(f"learning_rate_{exp_id}", epoch_loss_sum / (i_log + 1), epoch * len(train_data_loader) + i_log)

    # if epoch%save_steps == 0:
    #     torch.save(model.state_dict(), save_head_to(epoch))
    # scheduler1.step()
    # scheduler2.step()

# %%
writer.close()
