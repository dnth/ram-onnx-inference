import os
import time

import cupy as cp
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def transforms(images):
    transformed_images = []
    for image in images:
        image = image.convert("RGB")
        image = image.resize((384, 384), Image.BILINEAR)
        img_cp = cp.asarray(image).astype(cp.float32) / 255.0
        img_cp = cp.transpose(img_cp, (2, 0, 1))
        transformed_images.append(img_cp)

    batch = cp.stack(transformed_images)
    mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(1, 3, 1, 1)
    std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(1, 3, 1, 1)
    batch = (batch - mean) / std

    return cp.asnumpy(batch).astype(np.float32)


def postprocess(output, tag_list, tag_list_chinese):
    tags = cp.asarray(output[0])
    tag_english = []
    tag_chinese = []
    for b in range(tags.shape[0]):
        index = cp.asnumpy(cp.argwhere(tags[b] == 1).flatten())
        token = np.array(tag_list)[index]
        tag_english.append(token.tolist())
        token_chinese = np.array(tag_list_chinese)[index]
        tag_chinese.append(token_chinese.tolist())
    return tag_english, tag_chinese


def load_tag_lists(english_file, chinese_file):
    with open(english_file, "r") as f:
        tag_list = [line.strip() for line in f.readlines()]
    with open(chinese_file, "r", encoding="utf-8") as f:
        tag_list_chinese = [line.strip() for line in f.readlines()]
    return tag_list, tag_list_chinese


def create_onnx_session(model_path, providers):
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def warm_up_session(session, input_name, output_name):
    dummy_input = np.random.randn(1, 3, 384, 384).astype(np.float32)
    session.run([output_name], {input_name: dummy_input})


def process_images(
    image_paths,
    session,
    input_name,
    output_name,
    tag_list,
    tag_list_chinese,
    batch_size,
):
    results = []
    total_latency = 0

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch = image_paths[i : i + batch_size]
        images = [Image.open(path) for path in batch]

        transformed_batch = transforms(images)

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


def process_folder(
    folder_path,
    session,
    input_name,
    output_name,
    tag_list,
    tag_list_chinese,
    batch_size,
):
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    print(f"Found {len(image_files)} images in the folder.")
    results, total_latency = process_images(
        image_files,
        session,
        input_name,
        output_name,
        tag_list,
        tag_list_chinese,
        batch_size,
    )
    avg_latency = total_latency / len(results) if results else 0

    df = pd.DataFrame(results)
    parquet_file = f"{folder_path.replace('/', '_')}_results.parquet"
    df.to_parquet(parquet_file)

    print(f"Results saved to {parquet_file}")

    return results, avg_latency, parquet_file


if __name__ == "__main__":
    # Configuration
    folder_path = "sample_images"
    batch_size = 1
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

    # Load tag lists
    tag_list, tag_list_chinese = load_tag_lists(
        "data/ram_tag_list.txt", "data/ram_tag_list_chinese.txt"
    )

    # Create and warm up ONNX session
    session, input_name, output_name = create_onnx_session(model_path, providers)
    warm_up_session(session, input_name, output_name)

    # Process folder
    results, avg_latency, parquet_file = process_folder(
        folder_path,
        session,
        input_name,
        output_name,
        tag_list,
        tag_list_chinese,
        batch_size,
    )

    print(f"Processed {len(results)} images")
    print(f"Average inference latency: {avg_latency:.2f} ms")
    print(f"Results saved to {parquet_file}")
