import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, image_size=384):
        self.image_size = tuple(image_size) # 新版torch要求必须时tuple
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir

        self.rgb_files, self.depth_files = self.load_file_paths()
        self.check_files_exist()
        print("数据检查")
        for i in range(3):
            print(self.rgb_files[i])
            print(self.depth_files[i])
            print()

    def load_file_paths(self):
        rgb_files, depth_files = [], []

        for root, dirs, files in os.walk(self.rgb_dir):
            for rgb_file in files:
                rgb_file_path = os.path.join(root, rgb_file)
                relative_path = os.path.relpath(rgb_file_path, self.rgb_dir)
                depth_file = relative_path.replace("rgb", "depth_zbuffer")
                depth_file_path = os.path.join(self.depth_dir, depth_file)
                rgb_files.append(rgb_file_path)
                depth_files.append(depth_file_path)

        return rgb_files, depth_files

    def check_files_exist(self):
        for rgb, depth in zip(self.rgb_files, self.depth_files):
            if not os.path.exists(rgb) or not os.path.exists(depth):
                raise FileNotFoundError(f"One or more files do not exist:\nRGB: {rgb}\nDepth: {depth}")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.rgb_files[idx]
        depth_file = self.depth_files[idx]

        trans_totensor_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=0.5, std=0.5),
        ])

        trans_totensor_depth = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
        ])

        rgb = self.read_rgb_file(rgb_file, trans_totensor_rgb)
        depth = self.read_depth_file(depth_file, trans_totensor_depth)

        return rgb, depth

    def read_rgb_file(self, file_path, transform):
        img = Image.open(file_path)
        img_tensor = transform(img)
        return img_tensor

    def read_depth_file(self, file_path, transform):
        depth = Image.open(file_path)
        depth_tensor = transform(depth)
        depth_tensor[depth_tensor > 23000] = 0
        depth_tensor = depth_tensor / 512
        return depth_tensor
