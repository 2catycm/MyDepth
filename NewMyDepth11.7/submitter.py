#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
import sys
sys.path.append((this_directory.parent).as_posix())
project_directory = this_directory.parent

intput_picture_directory = this_directory/'test_data_a'
output_picture_directory = this_directory/'output'
output_picture_directory.mkdir(exist_ok=True, parents=True)

# 选择网络结构
from models import *
head = MyNetwork_large()

# 选择模型权重
pretrained_head = 'best_model/MyNetwork_large_1_2600_0.1744.pth'
pretrained_weights_path = 'pretrained_models/omnidata_dpt_depth_v2.ckpt'


#%%
import zipfile
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from modules.midas.dpt_depth import DPTDepthModel
from pathlib import Path
from PIL import Image
import tqdm
import torch.nn.functional as F

# 跳过报错图片
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DepthDataset(torch.utils.data.Dataset):
    """Some Information about DepthDataset"""
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.image_paths = list(input_dir.glob('*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, image_path
    def collate_fn(self, batch):
        images = []
        paths = []
        for image, path in batch:
            images.append(image)
            paths.append(path)

        # 将图像张量和路径列表封装成一个元组
        return torch.stack(images), paths
    
import imageio
import numpy as np
def do_submit_batched(model, input_picture_directory, 
                      output_picture_directory, batch_size=4,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
    )
    
    dataset = DepthDataset(input_picture_directory, 
                                    transform=transform)
    # 放到gpu的原理：如果在dataset阶段直接放入，显存过大；
    # dataloader只是一个迭代器，没有to device的说法。pin_memory的机制是数据存到显存，但是不是那个意思。
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=False, 
                                            collate_fn=dataset.collate_fn)
    
    # 使用模型进行批量推理
    bar = tqdm.tqdm(dataloader, desc='[Inferencing]')
            
    post_transform = transforms.Resize((720, 1280))
    with torch.no_grad():
        for inputs, paths in bar:
            inputs = inputs.to(device)
            depths_batch = model(inputs)
            depths_batch = post_transform(depths_batch)

            for depth, path in zip(depths_batch, paths):
                file_name = Path(path).stem + '.png'

                zip_file_location = output_picture_directory/file_name
                # save_raw_16bit(depth, zip_file_location)
                depth = depth.squeeze().cpu().numpy()
                imageio.imwrite(zip_file_location, depth.astype(np.uint16))
            

# omni
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
omni = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint
omni.load_state_dict(state_dict)
omni = omni.to(device)

# head
head = head.to(device)
if pretrained_head is not None:
    checkpoint_head = torch.load(pretrained_head, map_location=map_location)
    head.load_state_dict(checkpoint_head,strict=True)
    
def model_method(x):
    # print(x.shape)
    feature_map, output = omni(x) #
    output = output.squeeze(dim=1)
    # output = output.clamp(min=0, max=1) #限制0到1之间
    # output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0) #双线性插值放大到512x512
    output = output.clamp(0, 1)
    relative_depth_map = output
    
    result = head(feature_map)           #result的size为4x2，4是batch_size
    scale, shift = result[:, 0], result[:, 1]
    scale = scale.unsqueeze(1).unsqueeze(2)  # 扩展 scale 的维度
    shift = shift.unsqueeze(1).unsqueeze(2)  # 扩展 shift 的维度
    #print(relative_depth_map.shape, scale.shape)
    absolute_depth_map = relative_depth_map * scale + shift

    return absolute_depth_map * 1000 # 比赛要求


#%%
do_submit_batched(model_method, intput_picture_directory, 
                  output_picture_directory, 
                   batch_size=4, 
                  device = device,
                  )
# %%
