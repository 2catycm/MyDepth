import os
from PIL import Image
from torchvision import transforms
import cv2
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, image_size=384):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.image_size = image_size

        #获取rgb文件路径
        rgb_files = []
        for root, dirs, files in os.walk(rgb_dir):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".PNG"):
                    file_path = os.path.join(root, filename)
                    rgb_files.append(file_path)
        self.rgb_files = rgb_files

        #获取depth文件路径
        depth_files = []
        for root, dirs, files in os.walk(depth_dir):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".PNG"):
                    file_path = os.path.join(root, filename)
                    depth_files.append(file_path)
        self.depth_files = depth_files

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.rgb_files[idx]
        depth_file = self.depth_files[idx]

        trans_totensor_rgb = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        trans_totensor_depth = transforms.Compose([
           #transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        rgb = self.read_rgb_file(rgb_file, trans_totensor_rgb)
        depth = self.read_depth_file(depth_file, trans_totensor_depth)

        return rgb, depth

    def read_rgb_file(self, file_path, transform):
        img = Image.open(file_path)
        img_tensor = transform(img)
        return img_tensor

    def read_depth_file(self, file_path, transform):
        depth = cv2.imread(file_path, -1)
        depth[depth > 23000] = 0
        depth = depth / 512
        depth = Image.fromarray(depth)
        depth = transform(depth).squeeze(0)
        return depth
