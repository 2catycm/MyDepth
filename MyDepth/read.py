import os
from PIL import Image
from torchvision import transforms
import cv2
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, image_size=384):
        self.image_size = image_size
        rgb_files, depth_files = [], []
        for files in os.listdir(rgb_dir):
            for rgb_file in os.listdir(os.path.join(rgb_dir, files)):
                rgb_file_path = os.path.join(rgb_dir, files, rgb_file)
                depth_file = rgb_file.replace("rgb", "depth_zbuffer")
                depth_file_path = os.path.join(depth_dir, files, depth_file)
                rgb_files.append(rgb_file_path)
                depth_files.append(depth_file_path)
        self.rgb_files = rgb_files
        self.depth_files = depth_files
        print(self.rgb_files[:5])
        print(self.depth_files[:5])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.rgb_files[idx]
        depth_file = self.depth_files[idx]

        trans_totensor_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

        trans_totensor_depth = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
            ]
        )

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