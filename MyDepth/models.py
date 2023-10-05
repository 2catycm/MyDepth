#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys
sys.path.append((project_directory/'omnidata/omnidata_tools/torch').as_posix())
sys.path.append((project_directory/'ZoeDepth').as_posix())
#%%
from MyCommons import *
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
    # map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
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
    res = model.core.core.load_state_dict(state_dict, strict=False)
    print(res)
    set_require_grad(model.core.core, False) # 冻结
    return model

def set_require_grad(model, require_grad=False):
    for param in model.parameters():
        param.requires_grad = require_grad
        

# 定义网络
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
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(32 * 12 * 12, 2)
        # self.fc = nn.Linear(8192 , 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x