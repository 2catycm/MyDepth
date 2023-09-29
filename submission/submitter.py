#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
import sys
sys.path.append((this_directory.parent/'ZoeDepth').as_posix())
sys.path.append((this_directory.parent).as_posix())

#%%
from PIL import Image
from ZoeDepth.zoedepth.utils.misc import save_raw_16bit
from utils import DepthMapPseudoColorize
def do_submit(model:callable, intput_picture_directory:Path, output_picture_directory:Path, visualize_picture_directory:Path=None):
    # 1. 遍历 intput_picture_directory
    for picture_path in intput_picture_directory.glob('*.jpg'):
        # 读取图片
        image = Image.open(picture_path)
        # 推理
        depth = model.infer_pil(image)
        # 保存
        file_name = picture_path.stem+'.png'
        out_file_path = output_picture_directory/(file_name)
        save_raw_16bit(depth, out_file_path)
        if visualize_picture_directory is not None:
            vis_file_path = visualize_picture_directory/(file_name)
            DepthMapPseudoColorize(out_file_path, vis_file_path)
    # 2. 将 output_picture_directory 打包为对应名称的zip
    import zipfile
    import os
    zip_file_path = this_directory/f'{output_picture_directory.name}.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        for file_name in output_picture_directory.glob('*.png'):
            zip_file.write(file_name, arcname=file_name.name)
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
#%%
do_submit(zoe, intput_picture_directory, output_picture_directory, visualize_picture_directory)
# %%
