import time

import numpy as np
import onnxruntime as ort
from PIL import Image


def transform_numpy(image):
    image = image.convert("RGB")
    image = image.resize((384, 384), Image.BILINEAR)
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    img_np = (img_np - mean) / std
    return img_np.astype(np.float32)


def postprocess(output, tag_list, tag_list_chinese):
    tags = output[0]
    tag_output = []
    tag_output_chinese = []
    for b in range(tags.shape[0]):
        index = np.argwhere(tags[b] == 1)
        token = np.array(tag_list)[index].squeeze(axis=1)
        tag_output.append(" | ".join(token))
        token_chinese = np.array(tag_list_chinese)[index].squeeze(axis=1)
        tag_output_chinese.append(" | ".join(token_chinese))
    return tag_output, tag_output_chinese


# Load tag lists
with open("data/ram_tag_list.txt", "r") as f:
    tag_list = [line.strip() for line in f.readlines()]
with open("data/ram_tag_list_chinese.txt", "r", encoding="utf-8") as f:
    tag_list_chinese = [line.strip() for line in f.readlines()]

# Load and preprocess the image
image_path = "image_test_ram.jpg"
image = Image.open(image_path)

transformed_image = transform_numpy(image)
transformed_image = np.expand_dims(transformed_image, axis=0)

# Create ONNX session
model_path = "ram.onnx"
provider = "TensorrtExecutionProvider"  # Change to "CPUExecutionProvider" if needed
session = ort.InferenceSession(model_path, providers=[provider])

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# warm up
session.run([output_name], {input_name: transformed_image})
session.run([output_name], {input_name: transformed_image})

start_time = time.time()
output = session.run([output_name], {input_name: transformed_image})
end_time = time.time()

inference_latency = (end_time - start_time) * 1000  # Convert to milliseconds

# Postprocess the output
english_tags, chinese_tags = postprocess(output, tag_list, tag_list_chinese)

# Print results
print("English tags:", english_tags)
print("Chinese tags:", chinese_tags)
print(f"Inference latency: {inference_latency:.2f} ms")
