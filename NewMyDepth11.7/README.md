# 代码说明



## 环境配置

​	如Dockerfile 中所示



## 数据

​	使用了官方提供的taskonomy数据集



## 预训练模型

​	使用了Omnidata提供的预训练模型(https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch#readme)

​	利用其提供的用于深度估计的预训练模型(v2版)来估计相对深度

​	运行sh ./tools/download_depth_models.sh即可下载得到omnidata_dpt_depth_v2.ckpt

​	MD5为1071266b8bfdb58cdcb490d6fcddedb0

​	另外，构建网络的时候，即

   model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid

​	还会自动下载jx_vit_base_resnet50_384-9fd3c705.pth预先训练模型

​	MD5为d9248f440afcf545468a0de132a75c03

​	这两个预训练模型的文件都放在了project/models/pretrained_models中



## 整体思路介绍

​	使用Omnidata提供的预训练模型获得相对深度，然后训练一个尺度分支网络，该网络以Omnidata提供的预训练模型得到的feature_map为输入，输出scale和shift，最后的绝对深度即 相对深度*scale+shift。



## 算法细节

​	Omnidata提供的预训练模型中，中间的表示layer_4_rn作为feature_map(modules/midas/dpt_depth.py中76行)，尺度分支网络就三层卷积+Relu，最后全连接得到两个输出(见models_large2.py)。



## 训练流程

​	固定Omnidata的预训练模型的参数，只训练尺度分支网络。学习率设置为1e-5，采用AdamW优化器。



## 复现说明

​	注意204829图片是损害的，但为了代码实现方便，我们的输出中依然包含了这一张图片的预测图，评测时请跳过这一张图片。



​	复现有问题请发邮件。



​	运行时请在团队目录下，即/data/PZsdgj-19/19_2catycm下，执行以下指令(把team和official_root代入，gpu型号设为3，可修改)

```
# 构建环境
docker build -t 19_2catycm .

# 推理阶段：运行下述命令后，应当在output目录下生成目标文件
docker run --rm -it --gpus "device=3" \
-v /data/official_train:/workspace/official_train \
-v /data/official_b:/workspace/official_b \
-v /data/PZsdgj-19/19_2catycm/project:/workspace/project \
19_2catycm /bin/bash ./test.sh

# 训练阶段：运行下述命令后，应当在 best_model 目录下生成产生排行榜成绩的模型文件
rm -f project/best_model/*
docker run --rm -it --gpus "device=3" \
-v /data/official_train:/workspace/official_train \
-v /data/official_b:/workspace/official_b \
-v /data/PZsdgj-19/19_2catycm/project:/workspace/project \
19_2catycm /bin/bash ./train.sh
```





​	





​	
