import onnx
import onnxruntime as ort
import torch
from onnxsim import simplify
from ram.models import ram
from timm.utils.model import reparameterize_model


def export_ram_to_onnx(
    model_path,
    output_path,
    image_size=384,
    device="cpu",
    quantize=False,
    simplify_model=False,
    batch_size=None,  # New argument
):
    # Initialize the model
    model = ram(
        pretrained=model_path,
        image_size=image_size,
        vit="swin_l",
    )
    model.eval()
    model = model.to(device)

    model = reparameterize_model(model)

    # Define custom forward function
    def custom_forward(self, image):
        label_embed = torch.nn.functional.relu(self.wordvec_proj(self.label_embed))
        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        bs = image_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode="tagging",
        )

        logits = self.fc(tagging_embed[0]).squeeze(-1)
        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(image.device),
            torch.ones_like(logits),  # Use ones_like to match logits shape
            torch.zeros_like(logits),  # Use zeros_like to match logits shape
        )

        return targets  # Return only the targets tensor

    # Replace the forward method
    model.forward = custom_forward.__get__(model)

    # Export to ONNX
    dummy_input = torch.randn(
        1 if batch_size is None else batch_size, 3, image_size, image_size
    )
    dynamic_axes = (
        {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        if batch_size is None
        else None
    )

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # Verify the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully.")

    if simplify_model:
        print("Simplifying ONNX model...")
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, output_path)
        print(f"Simplified model saved to {output_path}")

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_output_path = output_path.replace(".onnx", "_quantized.onnx")
        quantize_dynamic(
            output_path, quantized_output_path, weight_type=QuantType.QUInt8
        )
        print(f"Quantized model exported to {quantized_output_path}")

    print(f"Model exported to {output_path}")

    # Test inference with ONNX Runtime
    session = ort.InferenceSession(
        output_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: dummy_input.cpu().numpy()})
    print("ONNX Runtime inference test successful.")


if __name__ == "__main__":
    model_path = "ram_swin_large_14m.pth"
    output_path = "ram.onnx"
    export_ram_to_onnx(
        model_path, output_path, quantize=True, simplify_model=True, batch_size=None
    )
