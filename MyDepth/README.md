## git clone
git clone 

如果没有加载出omnidata和ZoeDepth
<!-- git submodule sync -->
<!-- git submodule update --init -->
## 配置虚拟环境
conda create -n depth python=3.10
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

## 下载omnidata预训练模型：
从 https://drive.google.com/uc?id=1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t 下载
到 omnidata/omnidata_tools/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt

<!-- ##  -->