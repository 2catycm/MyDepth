# 代码说明

## 环境配置
首先需要git clone
```bash
git clone https://github.com/2catycm/MyDepth.git
git submodule update --init --recursive
cd MyDepth/MyDepth # 
```
注意，遵循复赛规范的文件夹为MyDepth/MyDepth，但是需要外面的MyDepth下的其他依赖模块，因此需要git clone而不是直接使用submit.zip


接下来配置环境的方式如 Dockerfile 中所示，基于pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime镜像
使用清华pypi源安装依赖包。




## 数据

   使用了官方提供的taskonomy数据集和replica_full_plus数据集。

如果数据加载路径不对，可以在 MyDepth/MyDepth/boilderplate.py 中修改



## 预训练模型

   使用了Omnidata提供的预训练模型(https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch#readme)

   利用其提供的用于深度估计的预训练模型(v2版)来估计相对深度

   运行sh ./tools/download_depth_models.sh即可下载得到omnidata_dpt_depth_v2.ckpt， 检查 MD5为1071266b8bfdb58cdcb490d6fcddedb0

​    如果sh无法运行，可以从 https://drive.google.com/uc?id=1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t 下载。


​    **请将下载文件放到到 MyDepth/omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt 的位置**




   另外，构建网络的时候，即

   model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid

   还会自动下载jx_vit_base_resnet50_384-9fd3c705.pth预先训练模型

   MD5为d9248f440afcf545468a0de132a75c03



## 整体思路介绍

   使用Omnidata提供的预训练模型获得相对深度，
<!-- 然后训练一个尺度分支网络，该网络以Omnidata提供的预训练模型得到的feature_map为输入，输出scale和shift，最后的绝对深度即 相对深度*scale+shift。 -->
    然后使用DPT结构训练一个尺度网络和一个偏移网络。
    最后的绝对深度即 相对深度*scale+shift。


## 算法细节

   请见MyDepth/MyDepth/models中的ThreeDPT这个类。

​    为了结合先验信息，我们将 relative，scale和shift的输出都放到0-1之间，然后使用20作为最大深度，0.1作为最大偏移。
​    使用 output = (relative)*scale*20+shift*0.1



## 训练流程

   使用PEFT微调Omnidata的预训练模型的参数，以适应新的场景的相对深度。
    采用SGD优化器和SAM优化器，学习率等参数如 train_aggresive.py 所示

要想运行训练，复现B榜的提交，

跑 `python  train_aggresive.py`即可

## 推理流程

跑 `python  submitter.py`即可，如果有路径问题，可以修改该文件中的input_picture_directory变量



## 复现说明



   复现有问题请发邮件。





   





   
