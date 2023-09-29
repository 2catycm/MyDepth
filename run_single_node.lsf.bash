#!/bin/bash
##BSUB -q hgx
#BSUB -q ssc-gpu

#BSUB -n 64
#BSUB -R "span[ptile=64]" 
#BSUB -R "rusage[mem=50GB]"

##BSUB -gpu "num=4/host:mode=exclusive_process"
##BSUB -gpu "num=7/host:mode=exclusive_process"
##BSUB -gpu "num=8/host:mode=exclusive_process"
#BSUB -gpu "num=1/host:mode=exclusive_process"


#BSUB -e log/%J.err
#BSUB -o log/%J.out

# hostfile='/work/APAC-TY/feiyeung/ds_hostfile'
# echo $LSB_DJOB_HOSTFILE
hostfile=$LSB_DJOB_HOSTFILE
# hostfile=./ds_hostfile
NP=$(cat "$hostfile" | wc -l)
cat "$hostfile"
echo Number of Process is "$NP"
cd $LS_SUBCWD || exit

source ~/.bashrc
# module load cuda/11.8

spack load pdsh
# conda activate $env_name
which pdsh
pdsh -V
nvidia-smi

export PATH=~/miniconda3/envs/hf_ai/bin/:$PATH

env_name=base
# conda run -n $env_name make run-bloom
conda run -n $env_name make run-bloom-quantize

# target_exe=benchmarks/benchmark_latency.py
# batch_size=128
# model='facebook/opt-125m'
# model='/share/data/bloom'

# python=~/miniconda3/envs/vllm/bin/python

# # conda run -n $env_name python $target_exe \
# $python $target_exe \
# --model $model \
# --tensor-parallel-size 8 \
# --batch-size $batch_size \
# --num-iters 5 \
# --input-len 6 \
# --output-len 100 \
#  | tee $target_exe-hgx-bs=$batch_size.txt

# 加载时间很长，需要15min。
# 查看 ray dashboard 端口转发 
# ssh -N -L 8265:localhost:8265 b05u17g