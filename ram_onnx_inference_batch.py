import os
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def transform_numpy(images):
    transformed_images = []
    for image in images:
        image = image.convert("RGB")
        image = image.resize((384, 384), Image.BILINEAR)
        img_np = np.array(image).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)
        transformed_images.append(img_np)

    batch = np.stack(transformed_images)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    batch = (batch - mean) / std
    return batch.astype(np.float32)


def postprocess(output, tag_list, tag_list_chinese):
    tags = output[0]
    tag_english = []
    tag_chinese = []
    for b in range(tags.shape[0]):
        index = np.argwhere(tags[b] == 1).flatten()
        token = np.array(tag_list)[index]
        tag_english.append(token.tolist())
        # tag_english.append(" | ".join(token))
        token_chinese = np.array(tag_list_chinese)[index]
        tag_chinese.append(token_chinese.tolist())
        # tag_chinese.append(" | ".join(token_chinese))
    return tag_english, tag_chinese


# Load tag lists
with open("data/ram_tag_list.txt", "r") as f:
    tag_list = [line.strip() for line in f.readlines()]
with open("data/ram_tag_list_chinese.txt", "r", encoding="utf-8") as f:
    tag_list_chinese = [line.strip() for line in f.readlines()]

# Create ONNX session
model_path = "ram.onnx"
providers = [
    (
        "TensorrtExecutionProvider",
        {
            "device_id": 0,
            "trt_max_workspace_size": 8589934592,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_force_sequential_engine_build": False,
            "trt_max_partition_iterations": 10000,
            "trt_min_subgraph_size": 1,
            "trt_builder_optimization_level": 5,
            "trt_timing_cache_enable": True,
        },
    ),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


session = ort.InferenceSession(model_path, providers=providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# warm up
dummy_input = np.random.randn(1, 3, 384, 384).astype(np.float32)
session.run([output_name], {input_name: dummy_input})


def process_batch(image_paths, batch_size):
    results = []
    total_latency = 0

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch = image_paths[i : i + batch_size]
        images = [Image.open(path) for path in batch]

        transformed_batch = transform_numpy(images)

        start_time = time.time()
        output = session.run([output_name], {input_name: transformed_batch})
        end_time = time.time()

        english_tags, chinese_tags = postprocess(output, tag_list, tag_list_chinese)
        inference_latency = (end_time - start_time) * 1000  # Convert to milliseconds

        for j, path in enumerate(batch):
            results.append(
                {
                    "filename": os.path.basename(path),
                    "english_tags": english_tags[j],
                    "chinese_tags": chinese_tags[j],
                    "latency": inference_latency / len(batch),
                }
            )

        total_latency += inference_latency

    return results, total_latency


def process_folder(folder_path, batch_size):
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    print(f"Found {len(image_files)} images in the folder.")
    results, total_latency = process_batch(image_files, batch_size)
    avg_latency = total_latency / len(results) if results else 0

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)
    parquet_file = f"{folder_path.replace('/', '_')}_results.parquet"
    df.to_parquet(parquet_file)

    print(f"Results saved to {parquet_file}")

    return results, avg_latency, parquet_file


if __name__ == "__main__":
    folder_path = "sample_images"
    batch_size = 1
    results, avg_latency, parquet_file = process_folder(folder_path, batch_size)

    print(f"Processed {len(results)} images")
    print(f"Average inference latency: {avg_latency:.2f} ms")
    print(f"Results saved to {parquet_file}")
