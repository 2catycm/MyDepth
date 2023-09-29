#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
import sys
sys.path.append((this_directory/'ZoeDepth').as_posix())
#%%
import torch
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Zoe_N
model_zoe_n = torch.hub.load("./ZoeDepth", 
                            #  "ZoeD_N",
                             "ZoeD_NK",
                             source="local",
                             pretrained=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
zoe = model_zoe_n.to(device)
zoe
#%%
pretrained_weights_path = "/work/asc22/yecanming/researches/cv/DepthEstimation/omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt"
# map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
checkpoint = torch.load(pretrained_weights_path,
                        # map_location=map_location
                        map_location=torch.device('cpu')
                        )
if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
else:
    state_dict = checkpoint
#%%
state_dict.keys()
zoe.state_dict().keys()

def make_txt(name, keys):
    with open(name, 'w') as f:
        f.write("\n".join(sorted(keys)))

make_txt('omni_state_dict.txt' , state_dict.keys())
make_txt('zoe_state_dict.txt', zoe.state_dict().keys())

#%%
res = zoe.load_state_dict(state_dict, strict=False)
res
#%%
res.missing_keys
make_txt("missing_keys.txt", res.missing_keys)
#%%
# omnidata是512. zoe是384
res = zoe.core.core.load_state_dict(state_dict, strict=False)
res
# dir(zoe)
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html

# list(zoe.children().name)
#%%
# revised_omni_state_dict = 

#%%
# from torchsummary import summary
# summary(zoe.cpu(), (3, 224, 224), device='cpu')
#%%
# zoe_opt = torch.compile(zoe)
# %%

# %%
