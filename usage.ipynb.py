#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
import sys
sys.path.append((this_directory/'ZoeDepth').as_posix())
#%%
import torch
#%%
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Zoe_N
model_zoe_n = torch.hub.load("./ZoeDepth", 
                             "ZoeD_N",
                             source="local",
                             pretrained=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
zoe = model_zoe_n.to(device)
zoe

#%%


#%%
from ZoeDepth.zoedepth.utils.misc import get_image_from_url
URL = "https://pic4.zhimg.com/v2-139d89c1b71dec1f9064d36f9452df3f_r.jpg"
image = get_image_from_url(URL)
image
#%%
image.save("vis/test/input.png")
# %%
depth = zoe.infer_pil(image)
# %%
depth # ？单位是什么

#%%

# Save raw
from ZoeDepth.zoedepth.utils.misc import save_raw_16bit
fpath = "vis/test/output.png"
save_raw_16bit(depth, fpath)

# Colorize output
from ZoeDepth.zoedepth.utils.misc import colorize
from PIL import Image

colored = colorize(depth)

# save colored output
fpath_colored = "vis/test/output_colored.png"
Image.fromarray(colored).save(fpath_colored)
#%%
colored
depth
# Image.fromarray(depth)

# %%
# https://blog.csdn.net/OrdinaryMatthew/article/details/129040370
import numpy as np
import cv2
from PIL import Image

from utils import DepthMapPseudoColorize
DepthMapPseudoColorize(fpath,fpath_colored)


# %%
