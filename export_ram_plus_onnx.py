import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnxsim import simplify
from ram.models import ram_plus
from timm.utils.model import reparameterize_model


def export_ram_plus_to_onnx(
    model_path,
    output_path,
    image_size=384,
    device="cpu",
    quantize=False,
    simplify_model=False,
    batch_size=None,
):
    # Initialize the model
    model = ram_plus(
        pretrained=model_path,
        image_size=image_size,
        vit="swin_l",
    )
    model.eval()
    model = model.to(device)

    model = reparameterize_model(model)

    # Define custom forward function
    def custom_forward(self, image):
        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]

        des_per_class = int(self.label_embed.shape[0] / self.num_class)

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(
            dim=-1, keepdim=True
        )
        reweight_scale = self.reweight_scale.exp()
        logits_per_image = reweight_scale * image_cls_embeds @ self.label_embed.t()
        logits_per_image = logits_per_image.view(bs, -1, des_per_class)

        weight_normalized = F.softmax(logits_per_image, dim=2)
        label_embed_reweight = (
            torch.empty(bs, self.num_class, 512).to(image.device).to(image.dtype)
        )

        for i in range(bs):
            reshaped_value = self.label_embed.view(-1, des_per_class, 512)
            product = weight_normalized[i].unsqueeze(-1) * reshaped_value
            label_embed_reweight[i] = product.sum(dim=1)

        label_embed = torch.nn.functional.relu(self.wordvec_proj(label_embed_reweight))

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
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.num_class).to(image.device),
        )

        return targets

    # Replace the forward method
    model.forward = custom_forward.__get__(model)

    # Export to ONNX
    dummy_input = torch.randn(
        1 if batch_size is None else batch_size, 3, image_size, image_size
    )
    dynamic_axes = (
        {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }
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
    output_names = [output.name for output in session.get_outputs()]
    session.run(output_names, {input_name: dummy_input.cpu().numpy()})
    print("ONNX Runtime inference test successful.")


# Usage example
if __name__ == "__main__":
    model_path = "data/ram_plus_swin_large_14m.pth"
    output_path = "ram_plus.onnx"
    export_ram_plus_to_onnx(
        model_path, output_path, quantize=True, simplify_model=True, batch_size=1
    )
