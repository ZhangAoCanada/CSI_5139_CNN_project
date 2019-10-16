# export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
# export LD_INCLUDE_PATH="/usr/local/include:$LD_INCLUDE_PATH"
export CUDA_HOME="/home/salam/cuda-10.0"
export PATH="/home/salam/cuda-10.0/bin:$PATH"
export CPATH="/home/salam/cuda-10.0/include"
export CUDNN_INCLUDE_DIR="/home/salam/cuda-10.0/include"
export CUDNN_LIB_DIR="/home/salam/cuda-10.0/lib64"

#export LD_LIBRARY_PATH="/home/zhangfeihu/anaconda3/lib:$LD_LIBRARY_PATH"
#export LD_INCLUDE_PATH="/home/zhangfeihu/anaconda3/include:$LD_INCLUDE_PATH"
#export CUDA_HOME="/home/work/cuda-9.2"
#export PATH="/home/zhangfeihu/anaconda3/bin:/home/work/cuda-9.2/bin:$PATH"
#export CPATH="/home/work/cuda-9.2/include"
#export CUDNN_INCLUDE_DIR="/home/work/cudnn/cudnn_v7/include"
#export CUDNN_LIB_DIR="/home/work/cudnn/cudnn_v7/lib64"
TORCH=$(python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))")
#echo $TORCH
cd libs/GANet
python3 setup.py clean
rm -rf build
python3 setup.py build
cp -r build/lib* build/lib

cd ../sync_bn
python3 setup.py clean
rm -rf build
python3 setup.py build
cp -r build/lib* build/lib
