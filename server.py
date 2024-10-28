from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import pillow_heif

app = Flask(__name__)
CORS(app)

# Load model
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
model.eval()

logging.basicConfig(level=logging.DEBUG)

@app.route('/extract-image', methods=['POST'])
def extract_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    app.logger.info("Image received: %s", image_file.filename)

    try:
        # Load the image, checking for HEIC format
        image_file.seek(0)  # Ensure the pointer is at the start of the file
        if image_file.filename.lower().endswith('.heic'):
            # Open HEIC image with pillow_heif and convert to RGB
            heif_image = pillow_heif.read_heif(image_file)
            image = Image.frombytes(
                heif_image.mode,
                heif_image.size,
                heif_image.data
            )
            image = image.convert("RGB")  # Ensure compatibility with OpenCV
        else:
            # Handle other formats with OpenCV
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image.")

            # Convert OpenCV BGR to RGB for consistency
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image) 

        # Image preprocessing
        inputs = preprocess_image(image)

        # Image inference
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state

        # Flatten embeddings for easier comparison later
        flatten_embeddings = embeddings.detach().cpu().numpy().flatten()  # Ensure it's a plain numpy array

        return jsonify({"embeddings": flatten_embeddings.tolist()})

    except Exception as e:
        app.logger.error("Error processing image: %s", e)
        return jsonify({"embeddings": [0]*(197*768)}), 500

def preprocess_image(image):
    image = image.resize((224, 224))
    inputs = processor(images=image, return_tensors="pt")
    return inputs

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
