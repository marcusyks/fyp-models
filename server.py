from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import numpy as np
import pillow_heif
import cv2
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import timm
from classes import IMAGENET2012_CLASSES

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load BLIP and ViT models only once on startup
def load_models():
    print('Loading models...')
    mobile_model = timm.create_model('tf_mobilenetv3_small_minimal_100.in1k', pretrained=True)
    data_config = timm.data.resolve_model_data_config(mobile_model)
    mobile_processor = timm.data.create_transform(**data_config, is_training=False)

    vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
    vit_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device).eval()
    print('Models loaded!')
    return mobile_model, mobile_processor, vit_model, vit_processor

mobile_model, mobile_processor, vit_model, vit_processor = load_models()
logging.basicConfig(level=logging.DEBUG)

# Configure thread pool for async processing
executor = ThreadPoolExecutor()

@app.route('/extract-image', methods=['POST'])
async def extract_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    app.logger.info("Processing image: %s", image_file.filename)

    try:
        # Process image loading and inference asynchronously
        image = await asyncio.to_thread(load_image, image_file)
        embeddings_task = asyncio.to_thread(infer_embeddings, image)
        keywords_task = asyncio.to_thread(infer_keywords, image)

        embeddings, keywords = await asyncio.gather(embeddings_task, keywords_task)
        print(keywords)
        return jsonify({"embeddings": embeddings, "keywords": keywords})
    except Exception as e:
        app.logger.error("Error processing image: %s", e)
        return jsonify({"error": "Image processing failed"}), 500

def infer_keywords(image):
    with torch.no_grad():
        output = mobile_model(mobile_processor(image).unsqueeze(0))  # unsqueeze single image into batch of 1
        _, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

    predicted_classes = top5_class_indices[0].tolist()
    class_ids = list(IMAGENET2012_CLASSES.keys())
    keyword_list = []
    for idx in predicted_classes:
        class_id = class_ids[idx]  # Get class ID using index
        class_name = IMAGENET2012_CLASSES[class_id]  # Look up class name
        print(f"Predicted Index: {idx}, Class Name: {class_name}")
        keyword_list.append(class_name)
    return keyword_list

def infer_embeddings(image):
    inputs = vit_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = vit_model(**inputs).last_hidden_state
    return embeddings.cpu().numpy().flatten().tolist()  # Flatten at return

def load_image(image_file):
    image_file.seek(0)
    if image_file.filename.lower().endswith('.heic'):
        heif_image = pillow_heif.read_heif(image_file)
        return Image.frombytes(heif_image.mode, heif_image.size, heif_image.data).convert("RGB")
    else:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image.")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
