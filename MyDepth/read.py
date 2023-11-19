import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, 
                 image_size=384, do_fisheye_transform=True, 
                 replace_dict:dict = None
                 ):
        if replace_dict is None:
            replace_dict = {"rgb":"depth_zbuffer"}
        self.replace_dict = replace_dict
        self.do_fisheye_transform = do_fisheye_transform
        self.inv_maps = None
        self.image_size = tuple(image_size)  # 新版torch要求必须时tuple
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
                depth_file = relative_path
                if self.replace_dict is not None:
                    for k,v in self.replace_dict.items():
                        depth_file = depth_file.replace(k,v)
                depth_file_path = os.path.join(self.depth_dir, depth_file)
                rgb_files.append(rgb_file_path)
                depth_files.append(depth_file_path)

        return rgb_files, depth_files

    def check_files_exist(self):
        for rgb, depth in zip(self.rgb_files, self.depth_files):
            if not os.path.exists(rgb) or not os.path.exists(depth):
                raise FileNotFoundError(
                    f"One or more files do not exist:\nRGB: {rgb}\nDepth: {depth}"
                )

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.rgb_files[idx]
        depth_file = self.depth_files[idx]
# https://pytorch.org/vision/main/generated/torchvision.transforms.CenterCrop.html
        trans_totensor_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                # transforms.CenterCrop(self.image_size),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

        trans_totensor_depth = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                # transforms.CenterCrop(self.image_size), # center crop也无法避免有黑边，干脆要求有黑边。
            ]
        )
        import random
        do_fisheye_bool  = (random.random() < float(self.do_fisheye_transform))
            
        rgb, cropper = self.read_rgb_file(rgb_file, trans_totensor_rgb, do_fisheye_bool)
        depth = self.read_depth_file(depth_file, trans_totensor_depth, cropper, do_fisheye_bool)

        return rgb, depth

    def read_rgb_file(self, file_path, transform, do_fisheye_bool):
        img = cv2.imread(file_path)
        cropper = None
        if do_fisheye_bool:
            img = self.transform_to_fisheye(img)
            cropper = self.get_cropper(img)
            img = cropper(img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img)
        return img_tensor, cropper

    def read_depth_file(self, file_path, transform, cropper, do_fisheye_bool):
        # depth = Image.open(file_path)
        # depth = cv2.imread(file_path, -1)
        # depth_tensor = transform(depth)
        # depth_tensor[depth_tensor > 23000] = 0
        # depth_tensor = depth_tensor / 512
        # return depth_tensor
        depth = cv2.imread(file_path, -1)
        if do_fisheye_bool:
            depth = self.transform_to_fisheye(depth)
            depth = cropper(depth)
        depth[depth > 23000] = 0
        depth = depth / 512
        depth = Image.fromarray(depth)
        depth = transform(depth).squeeze(0)
        return depth

    def compute_inv_maps(self):
        # 相机参数载入
        fx = 549.0869836743506
        fy = 549.0393143358732
        cx = 664.8310221457081
        cy = 368.30351397130175
        k1 = -0.037032730499053215
        k2 = -9.331683195791314e-05
        k3 = -0.0025427846701313096
        k4 = 0.0005759176479469663
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, k3, k4])
        # 坐标点
        coordinates = np.zeros((int(self.image_size[0]), int(self.image_size[1]), 2), dtype=np.float32)
        for i in range(coordinates.shape[0]):
            for j in range(coordinates.shape[1]):
                coordinates[i, j, :] = [i, j]
        maps = cv2.fisheye.undistortPoints(
            coordinates, K=camera_matrix, D=dist_coeffs, R=np.eye(3), P=camera_matrix
        )
        inv_map1, inv_map2 = maps[:, :, 1], maps[:, :, 0]
        inv_map1, inv_map2 = cv2.convertMaps(
            inv_map1, inv_map2, dstmap1type=cv2.CV_16SC2
        )
        self.inv_maps = (inv_map1, inv_map2)
    def get_cropper(self, cv_array_bgr):
        # 3. 黑边外接矩形
        gray = cv2.cvtColor(cv_array_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        # 计算目标轮廓的最小外接矩形的坐标和尺寸
        x, y, w, h = cv2.boundingRect(cnt)
        # 裁剪图像为无黑边的矩形
        # crop = cv_array_bgr[y:y+h, x:x+w]
        return lambda cv_array_bgr:cv_array_bgr[y:y+h, x:x+w]
    
    def transform_to_fisheye(self, cv_array):
        # 返回的也是opencv格式的array
        # 1. pinhole undistort
        # 2. fisheye distort
        if self.inv_maps is None:
            self.compute_inv_maps()
        distorted = cv2.remap(
            cv_array,
            self.inv_maps[0],
            self.inv_maps[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        
        return distorted