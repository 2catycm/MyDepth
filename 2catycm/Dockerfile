# FROM pytorch/pytorch:1.9.0-cuda11.7-cudnn8-runtime
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY ./requirements.txt ./requirements.txt
COPY ./test.sh ./test.sh
COPY ./train.sh ./train.sh

RUN pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  