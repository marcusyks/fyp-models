from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, AutoModel
import numpy as np
import pillow_heif
import cv2
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load BLIP and ViT models only once on startup
def load_models():
    print('Loading models...')
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()
    vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
    vit_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device).eval()
    print('Models loaded!')
    return blip_model, blip_processor, vit_model, vit_processor

blip_model, blip_processor, vit_model, vit_processor = load_models()
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
        words = keywords.split()
        keywords = [word.strip() for word in words if word.lower() not in stop_words]
        print(keywords)
        return jsonify({"embeddings": embeddings, "keywords": keywords})
    except Exception as e:
        app.logger.error("Error processing image: %s", e)
        return jsonify({"error": "Image processing failed"}), 500

def infer_keywords(image, max_length=10):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_length=max_length, num_beams=5)
    return blip_processor.decode(output[0], skip_special_tokens=True)

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
