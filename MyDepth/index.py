import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import torch.onnx
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

def create_model():
    # path needs to be changed!!!
    path = "/home/ysn/Desktop/MyDepth/omnidata/omnidata_tools/torch/omnidata_dpt_depth_v2.ckpt"
    model = models.ThreeDPT(path).to(device)
    model.relative = model.relative.merge_and_unload()
    tensor_x = torch.rand((1, 3, 224, 224), dtype=torch.float32)
    export_output = torch.onnx.dynamo_export(model, tensor_x)
    export_output.save("Model.onnx")
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = total_params * 4 / (1024 ** 2)  # assuming 4 bytes per float32 parameter
    print(f"Total storage size of model parameters: {total_size_mb:.2f} MB")
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = create_model()

def pred(net, input_img):
    img_tensor = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
    )
    img_tensor = transform(img_tensor)
    img_tensor =img_tensor.unsqueeze(0)
    # print(img_tensor.shape)

    with torch.no_grad():
        depths = net(img_tensor)["metric_depth"]*1000
        post_transform = transforms.Resize((720, 1280))
        depths = post_transform(depths)
        depths = depths.squeeze().cpu().numpy()
    return depths

input_img = cv2.imread('img_input.jpg')
output_img = pred(net, input_img) # 以batch_size=1的配置推理深度图并返回
cv2.imwrite('img_output.png', output_img.astype(np.uint16))



