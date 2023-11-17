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


seed = 1
seed_torch(seed)
#%%
system_data_path = Path("/data/projects/depth").resolve()

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

# OmniScale （MyNetworkLarge）
batch_size = int(
    n_gpus * (single_gpu_memory * 1 / 5) * 0.95 * 1.1 * 24564 / 16334 * 24564 / 14322/4 *24564/17530 *24564/2800
)  
# 定义训练参数，方便调参
# lr = 3e-6
lr = 1e-5
# lr = 3e-4
# lr = 1e-3
# lr = 1e-2
# lr = 1e-1 # ZoeDepthOmni可用
# lr = 0.5
# lr = 0.9

lr = lr * batch_size/8

print(f"batch_size={batch_size}, learning_rate={lr}")
# save_epoch = 4
save_steps = 120 * 3
log_steps = 30
# num_epochs = 2
num_epochs = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")