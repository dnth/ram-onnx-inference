# ONNX Runtime with CUDA Setup Guide

This guide provides steps to set up ONNX Runtime with CUDA support using Conda.

## 1. Install CUDA components
Install CUDA 12.2.2 and related tools:

conda install -y -c nvidia cuda=12.2.2 cuda-tools=12.2.2 cuda-toolkit=12.2.2 cuda-version=12.2 cuda-command-line-tools=12.2.2 cuda-compiler=12.2.2 cuda-runtime=12.2.2

## 2. Install cuDNN
Install cuDNN 9.2.1.18:

conda install cudnn==9.2.1.18

## 3. Install ONNX Runtime GPU
Install ONNX Runtime with GPU support:

pip install -U onnxruntime-gpu==1.19.2

## 4. Install TensorRT
Install TensorRT and its dependencies:

pip install tensorrt==10.1.0 tensorrt-cu12==10.1.0 tensorrt-cu12-bindings==10.1.0 tensorrt-cu12-libs==10.1.0

## 5. Set up library paths
Add the Conda environment's library path and TensorRT library path to LD_LIBRARY_PATH:

export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/onnx-gpu/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/onnx-gpu/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH"

Note: Adjust the paths according to your Conda environment location.


## Results
```bash
Using TensorrtExecutionProvider:
Performing warmup inference...
Warmup inference completed (10 iterations).


TensorrtExecutionProvider results:
English tags (TensorrtExecutionProvider): ['breakfast | bulletin board | table | dinning table | plate | eat | food | fork | French toast | juice | man | omelet | orange juice | pancake | platter | sit | syrup | waffle']
Chinese tags (TensorrtExecutionProvider): ['早餐 | 公告栏 | 桌子/表格 | 餐桌 | 盘子 | 吃 | 食物 | 餐叉 | 法式吐司 | 果汁 | 男人 | 煎蛋卷 | 橙汁 | 煎饼 | 大浅盘 | 坐/放置/坐落 | 糖浆 | 华夫饼干']
Runtime per image (TensorrtExecutionProvider): 30.08 ms

Using CPUExecutionProvider:
Performing warmup inference...
Warmup inference completed (10 iterations).


CPUExecutionProvider results:
English tags (CPUExecutionProvider): ['breakfast | bulletin board | table | dinning table | plate | eat | food | fork | French toast | juice | man | omelet | orange juice | pancake | platter | sit | syrup | waffle']
Chinese tags (CPUExecutionProvider): ['早餐 | 公告栏 | 桌子/表格 | 餐桌 | 盘子 | 吃 | 食物 | 餐叉 | 法式吐司 | 果汁 | 男人 | 煎蛋卷 | 橙汁 | 煎饼 | 大浅盘 | 坐/放置/坐落 | 糖浆 | 华夫饼干']
Runtime per image (CPUExecutionProvider): 693.16 ms
2024-09-05 13:49:43.644540898 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2024-09-05 13:49:43.644569283 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.

Using CUDAExecutionProvider:
Performing warmup inference...
Warmup inference completed (10 iterations).


CUDAExecutionProvider results:
English tags (CUDAExecutionProvider): ['breakfast | bulletin board | table | dinning table | plate | eat | food | fork | French toast | juice | man | omelet | orange juice | pancake | platter | sit | syrup | waffle']
Chinese tags (CUDAExecutionProvider): ['早餐 | 公告栏 | 桌子/表格 | 餐桌 | 盘子 | 吃 | 食物 | 餐叉 | 法式吐司 | 果汁 | 男人 | 煎蛋卷 | 橙汁 | 煎饼 | 大浅盘 | 坐/放置/坐落 | 糖浆 | 华夫饼干']
Runtime per image (CUDAExecutionProvider): 31.51 ms
```