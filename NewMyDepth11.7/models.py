import torch.nn as nn
import torch

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.name = 'MyNetwork'
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(32 * 12 * 12, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MyNetwork_large(nn.Module):
    def __init__(self):
        super(MyNetwork_large, self).__init__()
        self.name = 'MyNetwork_large'
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(256 * 12 * 12, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MyNetwork_large_bn(nn.Module):
    def __init__(self):
        super(MyNetwork_large_bn, self).__init__()
        self.name = 'MyNetwork_large_bn'
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)  
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)  
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(256 * 12 * 12, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MyNetwork_large_384(nn.Module):
    def __init__(self):
        super(MyNetwork_large_384, self).__init__()
        self.name = 'MyNetwork_large_384'
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2)
        x = nn.functional.interpolate(x, size=[384, 384], mode='nearest')
        return x
    
import timm

#第一种改法    
class ResNet_v1(nn.Module):
    def __init__(self):
        super(ResNet_v1, self).__init__()
        self.name = 'ResNet18_v1'
        self.conv = nn.Conv2d(256, 3, kernel_size=1)
        self.resnet = timm.create_model('resnet18', pretrained=True)
        # self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # 修改最后一层全连接层以适应输出为2维
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        return x

#第二种改法   
class ResNet_v2(nn.Module):
    def __init__(self):
        super(ResNet_v2, self).__init__()
        self.name = 'ResNet18_v2'
        self.resnet = timm.create_model('resnet18', pretrained=True)
        # Modify the first convolutional layer to accept [8, 256, 12, 12] input
        self.resnet.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the last fully connected layer to output 2 features
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        self.name = 'U_Net'
        self.conv0 = nn.Conv2d(256, 3, kernel_size=1, stride=1)
        self.u_net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
        self.conv1 = nn.Conv2d(1, 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.u_net(x)
        x = self.conv1(x)
        return x