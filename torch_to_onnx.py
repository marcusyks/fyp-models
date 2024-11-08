# Load models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()
vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
vit_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device).eval()

# Convert BLIP model to ONNX
def convert_blip_to_onnx(blip_model, processor, output_path="blip_model.onnx"):
    dummy_input = processor("A sample image for captioning", return_tensors="pt").pixel_values
    torch.onnx.export(blip_model, dummy_input, output_path, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"},
                                    "output": {0: "batch_size", 1: "sequence_length"}})
    print(f"BLIP model saved as {output_path}")

# Convert ViT model to ONNX
def convert_vit_to_onnx(vit_model, processor, output_path="vit_model.onnx"):
    dummy_input = processor(images=np.random.rand(1, 3, 224, 224), return_tensors="pt").pixel_values
    torch.onnx.export(vit_model, dummy_input, output_path, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"},
                                    "output": {0: "batch_size", 1: "embedding_dimension"}})
    print(f"ViT model saved as {output_path}")

convert_blip_to_onnx(blip_model, blip_processor)
convert_vit_to_onnx(vit_model, vit_processor)
