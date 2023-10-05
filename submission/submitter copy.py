#%%
from pathlib import Path

from tqdm import tqdm

this_file = Path(__file__).resolve()
this_directory = this_file.parent
import sys
sys.path.append((this_directory.parent/'ZoeDepth').as_posix())
sys.path.append((this_directory.parent).as_posix())

#%%
from PIL import Image
from ZoeDepth.zoedepth.utils.misc import save_raw_16bit
from utils import DepthMapPseudoColorize
import zipfile
import os
# from ZoeDepth.zoedepth.data import transforms
from ZoeDepth.zoedepth.utils import misc
from torchvision.transforms import ToTensor
from torchvision import transforms
def do_submit(model:callable, input_picture_directory:Path, 
              output_picture_directory:Path, visualize_picture_directory:Path=None, 
              batch_size=128):
    pictures = list(input_picture_directory.glob('*.jpg'))
    zip_file_path = this_directory/f'{output_picture_directory.name}.zip'
    if visualize_picture_directory is not None:
        vis_file_path = this_directory/f'{visualize_picture_directory.name}.zip'
    # 1. 使用 Pytorch 的 DataLoader， 将 pictures 变成 tensor的batch
    # ToTensor()(img).
    transform = transforms.Compose([Image.open, ToTensor()])
    dataloader = torch.utils.data.DataLoader(pictures, 
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=4, 
                                             pin_memory=True, 
                                             collate_fn=transform)
    
    # 2. 使用 model 预测深度图
    model.eval()
    with torch.no_grad():
        bar = tqdm(enumerate(dataloader))
        for i, img in bar:
            img = img.to(device)
            depth = model(img)
            depth = depth.cpu()
            for j in range(depth.shape[0]):
                save_raw_16bit(depth[j], output_picture_directory/f'{i*batch_size+j:04d}.png')
                if visualize_picture_directory is not None:
                    colorized = DepthMapPseudoColorize()(depth[j])
                    colorized.save(visualize_picture_directory/f'{i*batch_size+j:04d}.png')
    # 3. 将深度图加入到zip文件中
    # 4. 解压zip文件，查看结果
  
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Zoe_N
model_zoe_n = torch.hub.load("../ZoeDepth", 
                            #  "ZoeD_N",
                             "ZoeD_NK",
                             source="local",
                             pretrained=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
zoe = model_zoe_n.to(device)
zoe
# %%
# 
intput_picture_directory = this_directory/'preliminary_a'
output_picture_directory = this_directory/'result'
output_picture_directory.mkdir(exist_ok=True, parents=True)
visualize_picture_directory = this_directory/'vis'
visualize_picture_directory.mkdir(exist_ok=True, parents=True)

#%%
visualize_picture_directory = None
do_submit(zoe, intput_picture_directory, output_picture_directory, 
          visualize_picture_directory)
# %%
