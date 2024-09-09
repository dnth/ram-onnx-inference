import os
import time
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from PIL import Image
from tqdm.auto import tqdm


class RecognizeAnythingModel:
    def __init__(self):
        from ram import get_transform
        from ram import inference_ram as inference
        from ram.models import ram

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.transform = get_transform(image_size=384)
        self.inference = inference

        logger.info("Loading model...")
        model_path = "data/ram_swin_large_14m.pth"

        model = ram(
            pretrained=model_path,
            image_size=384,
            vit="swin_l",
        )
        model.eval()
        self.model = model.to(self.device)

        logger.info("Model loaded successfully")

    def tag_image(self, image_path):
        image = self.transform(Image.open(image_path)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            english_tags, chinese_tags = self.inference(image, self.model)

        english_tags = [tag.strip() for tag in english_tags.split("|") if tag.strip()]
        chinese_tags = [tag.strip() for tag in chinese_tags.split("|") if tag.strip()]
        return english_tags, chinese_tags


def process_folder(folder_path):
    model = RecognizeAnythingModel()
    results = []
    total_inference_time = 0
    image_count = 0

    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                image_files.append(Path(root) / file)

    for image_path in tqdm(image_files, desc="Processing images", unit="image"):
        start_time = time.time()
        english_tags, chinese_tags = model.tag_image(image_path)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        image_count += 1

        results.append(
            {
                "filename": image_path.name,
                "english_tags": english_tags,
                "chinese_tags": chinese_tags,
            }
        )

    return pd.DataFrame(results), total_inference_time, image_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RAM model inference on images in a folder"
    )
    parser.add_argument(
        "folder_path", type=str, help="Path to the folder containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Path to save the results CSV file",
    )
    args = parser.parse_args()

    results_df, total_inference_time, image_count = process_folder(args.folder_path)

    # Save results to Parquet
    output_parquet = "pytorch_inference_results.parquet"
    results_df.to_parquet(output_parquet, index=False)
    print(f"Results saved to {output_parquet}")

    # Compute and display average inference latency
    if image_count > 0:
        avg_inference_latency = total_inference_time / image_count
        print(
            f"Average inference latency: {avg_inference_latency * 1000:.4f} ms per image"
        )
    else:
        print("No images were processed.")
