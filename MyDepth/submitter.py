#%%
from boilerplate import *
#%%

input_picture_directory = system_data_path/'5.跨场景单目深度估计/决赛数据/final_a'
output_picture_directory = project_directory/'result'
output_picture_directory.mkdir(exist_ok=True, parents=True)

# 选择网络结构
# from models import *
import models
model = models.OmniScale(pretrained_weights_path, head=models.MyNetwork_large())
model = model.to(device)

# 选择模型 state_dict
# pretrained_head = system_data_path/'runs/复现初赛-添加sam'/'OmniScale_2880.pth'
pretrained_head = system_data_path/'runs/复现初赛-添加sam'/'OmniScale_4320.pth'
checkpoint = torch.load(pretrained_head)
model.load_state_dict(checkpoint)

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
            # depths_batch = model(inputs)
            depths_batch = model(inputs)["metric_depth"]*1000
            depths_batch = post_transform(depths_batch)

            for depth, path in zip(depths_batch, paths):
                file_name = Path(path).stem + '.png'

                zip_file_location = output_picture_directory/file_name
                # save_raw_16bit(depth, zip_file_location)
                depth = depth.squeeze().cpu().numpy()
                imageio.imwrite(zip_file_location, depth.astype(np.uint16))


#%%
do_submit_batched(model, input_picture_directory, 
                  output_picture_directory, 
                   batch_size=batch_size, 
                  device = device,
                  )
# %%
