#%%
from boilerplate import *
# batch_size*=24564/8478*24564/7338*24564/12922*24564/21542
# batch_size*=24564/8478*24564/7338*24564/17842/3
batch_size*=24564/8478*24564/7338*24564/17842/3*24564/12160*0.8

batch_size=int(batch_size)
print(f"batch_size={batch_size}")
#%%

# input_picture_directory = system_data_path/'5.跨场景单目深度估计/决赛数据/final_a'
input_picture_directory = system_data_path/'5.跨场景单目深度估计/决赛数据/final_b'
output_picture_directory = project_directory/'result'
output_picture_directory.mkdir(exist_ok=True, parents=True)
#%%
# 选择网络结构
# from models import *
import models
# model = models.OmniScale(pretrained_weights_path, head=models.MyNetwork_large())
# model = models.ThreeDPT(pretrained_weights_path)
# model = model.to(device)

def load_into_model(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
# model1 = models.get_zoe_single_head_with_omni(pretrained_weights_path)

model1 = models.ThreeDPT(pretrained_weights_path)
# load_into_model(system_data_path/'runs/最激进-DPT-鱼眼优化'/'ThreeDPT_41480.pth', model1)
# load_into_model(system_data_path/'runs/最激进-DPT-鱼眼优化'/'ThreeDPT_42920.pth', model1)
load_into_model(system_data_path/'runs/最激进-DPT-鱼眼优化'/'ThreeDPT_44000.pth', model1)

# model2 = models.ThreeDPT(pretrained_weights_path)
# load_into_model(system_data_path/'runs/最激进'/'ThreeDPT_17579.pth', model2)

# model3 = models.ThreeDPT(pretrained_weights_path)
# load_into_model(system_data_path/'runs/3DPT稳定版-根据鱼眼做进一步微调'/'ThreeDPT_17480.pth', model3)
# model_to_weight = [model1, model2, model3]
# for model in model_to_weight:
#     model = model.to(device)

# model  = models.WeightedEnsemble(model_to_weight)
model = model1

model = model.to(device)

# 选择模型 state_dict
# pretrained_head = system_data_path/'runs/复现初赛-添加sam'/'OmniScale_2880.pth'
# pretrained_head = system_data_path/'runs/复现初赛-添加sam'/'OmniScale_4320.pth'
# pretrained_head = system_data_path/'runs/最激进'/'ThreeDPT_12960.pth' # 训练一轮的结果
# pretrained_head = system_data_path/'runs/最激进'/'ThreeDPT_13619.pth' # 训练一轮的结果
# pretrained_head = system_data_path/'runs/最激进'/'ThreeDPT_17579.pth' # 训练一轮的结果
# pretrained_head = system_data_path/'runs/3DPT稳定版-根据鱼眼做进一步微调'/'ThreeDPT_17480.pth' # 训练一轮的结果
# checkpoint = torch.load(pretrained_head)
# model.load_state_dict(checkpoint)

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
    model.eval() # 必须有
    
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
                                                # num_workers=1, 
                                                num_workers=4, 
                                                # num_workers=8, 
                                                # num_workers=12, 
                                                pin_memory=True, 
                                            collate_fn=dataset.collate_fn, 
                                            )
    
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
# 直接用linux命令行压缩
# zip -q -r result.zip result/*
# 检查文件数量
# 4301是对的
# ls -l result/ | wc -l
# zipdetails result.zip 这个命令没用
# zipinfo result.zip 挺好的命令

#%%
import os
from zipfile import ZipFile
import zipfile
import tqdm
def zip_folder(folder_path, output_zip):
    # 打开一个 Zip 文件，如果不存在则创建
    with ZipFile(output_zip, 'w', 
                #  compression=zipfile.ZIP_LZMA, 
                 compression=zipfile.ZIP_DEFLATED, # 比较快, 而且压缩地很好。
                 compresslevel=9) as zipf: # 9是压缩地最小的
        
        # 遍历文件夹中的所有文件
        for root, _, files in os.walk(folder_path):
            bar = tqdm.tqdm(files)
            for file in bar:
                file_path = os.path.join(root, file)

                # 将文件添加到 Zip 文件中，压缩文件内部没有层次结构
                zipf.write(file_path, os.path.basename(file_path))


# 打包文件夹
zip_folder(output_picture_directory.as_posix(), output_picture_directory.parent/f"{output_picture_directory.name}.zip")

# %%
