# ram-onnx-inference

This guide shows how to run the recognize-anything-model (RAM) with ONNX Runtime GPU.

## 1. Install CUDA components
Install CUDA 12.2.2 and related tools:

```bash
conda install -y -c nvidia cuda=12.2.2 cuda-tools=12.2.2 cuda-toolkit=12.2.2 cuda-version=12.2 cuda-command-line-tools=12.2.2 cuda-compiler=12.2.2 cuda-runtime=12.2.2
```

## 2. Install cuDNN
Install cuDNN 9.2.1.18:

```bash
conda install cudnn==9.2.1.18
```

## 3. Install ONNX Runtime GPU
Install ONNX Runtime with GPU support:

```bash
pip install -U onnxruntime-gpu==1.19.2
```

## 4. Install TensorRT
Install TensorRT and its dependencies:

```bash
pip install tensorrt==10.1.0 tensorrt-cu12==10.1.0 tensorrt-cu12-bindings==10.1.0 tensorrt-cu12-libs==10.1.0
```

## 5. Set up library paths
Add the Conda environment's library path and TensorRT library path to LD_LIBRARY_PATH:

```bash
export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/ram-onnx-inference/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/ram-onnx-inference/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH"
```

Note: Adjust the paths according to your Conda environment location.


Usage:

```bash
python ram_onnx_inference_batch.py [options]
```

Example:
```bash
python ram_onnx_inference_batch.py --folder_path /path/to/images --num_workers 4 --model_path /path/to/ram.onnx --output_file results.parquet
```

Options:
- `--folder_path`: Path to the folder containing images (default: "sample_images")
- `--num_workers`: Number of worker threads (default: 8)
- `--model_path`: Path to the ONNX model file (default: "ram.onnx")
- `--output_file`: Output file path for results (default: "onnx_inference_results.parquet")
