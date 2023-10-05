#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
import sys
sys.path.append((this_directory.parent/'ZoeDepth').as_posix())
sys.path.append((this_directory.parent).as_posix())
sys.path.append((this_directory.parent/"MyDepth").as_posix())
project_directory = this_directory.parent

import zipfile
import io
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from pathlib import Path
from ZoeDepth.zoedepth.utils.misc import save_raw_16bit
from PIL import Image
import tqdm
from matplotlib import pyplot as plt
import numpy as np
#%%
path = "result_1000/img_100001.png"
# path = "result/img_100001.png"
vis_path = "vis/img_100001.png"
image = Image.open(path)
image
# %%
tensor = transforms.ToTensor()(image)
tensor.shape
# scaled = (tensor-tensor.min())/(tensor.max()-tensor.min())*255
# scaled.shape
# transforms.ToPILImage()(scaled)
transforms.ToPILImage()(tensor/1000)
# %%
tensor.max(), tensor.min()
# %%
from utils import DepthMapPseudoColorize
DepthMapPseudoColorize(path,vis_path)
# %%
