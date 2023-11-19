#%%
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

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
zoe = model_zoe_n.to(device)
zoe
#%%
# from torchsummary import summary
# summary(zoe.cpu(), (3, 224, 224), device='cpu')
#%%
# zoe_opt = torch.compile(zoe)
# %%
import torch.onnx
import numpy as np

#Function to Convert to ONNX
def Convert_ONNX(model, input_size, path):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, *input_size, requires_grad=True)
    # dummy_input = np.random.randn(1, *input_size)
    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         path,       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')
import onnx
def check_onnx(path):
    onnx_model = onnx.load(path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")
        return onnx_model

# path = "myFirstModel.pth"
# model.load_state_dict(torch.load(path))

Convert_ONNX(zoe, (3, 224, 224), "ZoeD_N.onnx")
onnx_model = check_onnx("ZoeD_N.onnx")

# tensor_x = torch.rand((3, 224, 224), dtype=torch.float32)
# export_output = torch.onnx.dynamo_export(zoe, tensor_x)
# export_output.save("ZoeD_N.onnx")
# %%
