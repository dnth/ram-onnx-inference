import os
import time

import cupy as cp
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def transforms(image):
    image = image.convert("RGB")
    image = image.resize((384, 384), Image.BILINEAR)
    img_cp = cp.asarray(image).astype(cp.float32) / 255.0
    img_cp = cp.transpose(img_cp, (2, 0, 1))

    mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(3, 1, 1)
    std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(3, 1, 1)
    img_cp = (img_cp - mean) / std

    return cp.expand_dims(img_cp, axis=0)


def postprocess(output, tag_list, tag_list_chinese):
    tags = cp.asarray(output[0][0])
    index = cp.where(tags == 1)[0].get()
    tag_english = [tag_list[i] for i in index]
    tag_chinese = [tag_list_chinese[i] for i in index]
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

    for path in tqdm(image_paths, desc="Processing images"):
        image = Image.open(path)
        transformed_image = transforms(image)

        start_time = time.time()
        output = session.run([output_name], {input_name: cp.asnumpy(transformed_image)})
        end_time = time.time()

        english_tags, chinese_tags = postprocess(output, tag_list, tag_list_chinese)
        inference_latency = (end_time - start_time) * 1000  # Convert to milliseconds

        results.append(
            {
                "filename": os.path.basename(path),
                "english_tags": english_tags,
                "chinese_tags": chinese_tags,
                "latency": inference_latency,
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
