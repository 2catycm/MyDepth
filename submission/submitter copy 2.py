#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
import sys
sys.path.append((this_directory.parent/'ZoeDepth').as_posix())
sys.path.append((this_directory.parent).as_posix())

#%%
import zipfile
import io
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from pathlib import Path
from ZoeDepth.zoedepth.utils.misc import save_raw_16bit
from PIL import Image
import tqdm
# def process_image(model, image_tensor):
#     # 执行深度推理
#     depth = model(image_tensor.unsqueeze(0))  # 假设模型接受单个图像
#     return depth.squeeze()

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

def do_submit_batched(model, input_picture_directory, 
                      output_picture_directory, batch_size=4,
                     device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    # 准备一个内存中的ZIP文件
    output_zip_path = output_picture_directory.with_suffix('.zip')
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        transform = transforms.Compose([transforms.ToTensor()])
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
        model.eval()
        with torch.no_grad():
            for inputs, paths in bar:
                inputs = inputs.to(device)
                depths_batch = model.infer(inputs)
                
                for depth, path in zip(depths_batch, paths):
                    file_name = Path(path).stem + '.png'

                    zip_file_location = output_picture_directory/file_name
                    save_raw_16bit(depth, zip_file_location)
                    
# depths_batch = []
# for image_tensor in inputs:
# depth = process_image(model, image_tensor)
# depths_batch.append(depth)

# depths_batch = model(inputs)
    # depth_buffer = io.BytesIO()
    # save_raw_16bit(depth, depth_buffer)
    # 保存深度图像到内存中的ZIP文件
    #                 zip_file.writestr(file_name, depth_buffer.getvalue())

    # # 将内存中的ZIP文件保存到磁盘或返回给调用方，取决于需求
    # zip_buffer.seek(0)
    # with open(output_zip_path, 'wb') as f:
    #     f.write(zip_buffer.read())

# 直接用linux命令行压缩
# zip -q -r result.zip result/ 
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
# do_submit(zoe, intput_picture_directory, output_picture_directory, visualize_picture_directory)
do_submit_batched(zoe, intput_picture_directory, 
                  output_picture_directory, 
                #   batch_size=256, 
                  batch_size=64, 
                  device = device,
                  )
# %%
