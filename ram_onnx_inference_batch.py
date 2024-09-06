import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cupy as cp
import numpy as np
import onnxruntime as ort
import pandas as pd
from loguru import logger
from PIL import Image
from tqdm.auto import tqdm

# Constants for tag lists
TAG_LIST = [line.strip() for line in open("data/ram_tag_list.txt", "r")]
TAG_LIST_CHINESE = [
    line.strip()
    for line in open("data/ram_tag_list_chinese.txt", "r", encoding="utf-8")
]

# Configure Loguru
logger.add("ram_inference.log", rotation="10 MB")


def transforms(image):
    image = image.convert("RGB")
    image = image.resize((384, 384), Image.BILINEAR)
    img_cp = cp.asarray(image).astype(cp.float32) / 255.0
    img_cp = cp.transpose(img_cp, (2, 0, 1))

    mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(3, 1, 1)
    std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(3, 1, 1)
    img_cp = (img_cp - mean) / std

    return cp.expand_dims(img_cp, axis=0)


def postprocess(output):
    tags = cp.asarray(output[0][0])
    index = cp.where(tags == 1)[0].get()
    tag_english = [TAG_LIST[i] for i in index]
    tag_chinese = [TAG_LIST_CHINESE[i] for i in index]
    return tag_english, tag_chinese


def create_onnx_session(model_path, providers):
    logger.info(f"Creating ONNX session with model: {model_path}")
    session = ort.InferenceSession(model_path, providers=providers)
    logger.info("ONNX session created successfully")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logger.info(f"Input name: {input_name}, Output name: {output_name}")
    return session, input_name, output_name


def warm_up_session(session, input_name, output_name):
    logger.info("Warming up ONNX session")
    dummy_input = np.random.randn(1, 3, 384, 384).astype(np.float32)
    for _ in range(3):
        session.run([output_name], {input_name: dummy_input})


def process_single_image(path, session, input_name, output_name):
    image = Image.open(path)
    transformed_image = transforms(image)
    output = session.run([output_name], {input_name: cp.asnumpy(transformed_image)})
    english_tags, chinese_tags = postprocess(output)
    return {
        "filename": os.path.basename(path),
        "english_tags": english_tags,
        "chinese_tags": chinese_tags,
    }


def process_images(image_paths, session, input_name, output_name, num_workers):
    logger.info(f"Processing {len(image_paths)} images with {num_workers} workers")
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(
                process_single_image,
                path,
                session,
                input_name,
                output_name,
            ): path
            for path in image_paths
        }

        for future in tqdm(
            as_completed(future_to_path),
            total=len(image_paths),
            desc="Processing images",
        ):
            result = future.result()
            results.append(result)
    return results


if __name__ == "__main__":
    logger.info("Starting RAM ONNX inference batch process")

    # Configuration
    folder_path = "sample_images"
    num_workers = 16
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

    logger.info(f"Using providers: {providers}")

    # Create and warm up ONNX session
    logger.info("Initializing ONNX session...")
    session, input_name, output_name = create_onnx_session(model_path, providers)
    logger.info("ONNX session initialized")
    warm_up_session(session, input_name, output_name)

    # Get image files
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]
    logger.info(f"Found {len(image_files)} images in the folder: {folder_path}")

    # Process images
    start_time = time.time()
    results = process_images(
        image_files,
        session,
        input_name,
        output_name,
        num_workers,
    )
    end_time = time.time()

    total_inference_time = end_time - start_time
    avg_inference_time = total_inference_time / len(results) if results else 0

    # Save results to parquet
    df = pd.DataFrame(results)
    parquet_file = f"{folder_path.replace('/', '_')}_results.parquet"
    df.to_parquet(parquet_file)

    logger.info(f"Processed {len(results)} images")
    logger.info(f"Total inference time: {total_inference_time:.4f} seconds")
    logger.info(f"Average inference time per image: {avg_inference_time * 1000:.4f} ms")
    logger.info(f"Results saved to {parquet_file}")
