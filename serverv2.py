from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import numpy as np
import pillow_heif
import cv2
import logging
import io
import concurrent.futures
import mobileclip

"""Flask server to extract image embeddings and keywords using MobileCLIP."""

# Start the Flask app
app = Flask(__name__)

# Enable CORS for API calls
CORS(app)

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 100 keywords for classification
KEYWORDS = [
    "cat", "dog", "bird", "car", "mountain", "beach", "food", "person", "family", "holiday",
    "portrait", "landscape", "street", "city", "sky", "sunset", "sunrise", "park", "trees", "flowers",
    "home", "house", "office", "friends", "selfie", "travel", "vacation", "landmark", "event", "wedding",
    "birthday", "party", "baby", "children", "couple", "group", "night", "day", "outdoor", "indoor",
    "nature", "urban", "water", "ocean", "lake", "river", "mountain", "desert", "rain", "snow", "storm",
    "cloud", "sun", "cloudy", "clear", "moon", "stars", "celebration", "cooking", "meal", "restaurant",
    "food", "pizza", "burger", "sandwich", "sushi", "vegetables", "fruit", "coffee", "tea", "wine",
    "beer", "juice", "smoothie", "workout", "gym", "exercise", "running", "bike", "swimming", "sports",
    "basketball", "soccer", "baseball", "tennis", "volleyball", "fitness", "travel", "adventure", "mountain",
    "desert", "snowboarding", "skiing", "cycling", "hiking", "climbing", "relaxing", "home decor", "garden",
    "interior", "furniture", "clothing", "fashion", "shoes", "hat", "jewelry", "watch", "accessories",
    "art", "painting", "drawing", "sculpture", "book", "library", "school", "university", "graduation", "study"
]


def load_models():
    """Load the MobileCLIP model, image preprocessing function and encoded text features

    Returns:
        nn.Module: MobileCLIP instantiated model
        Any: Image preprocessing function
        Tensor: Encoded text features
    """
    app.logger.info("Loading models...")

    # Load MobileCLIP model and image preprocessing function
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s1')

    # Load MobileCLIP text tokenizer and encode the 100 keywords
    tokenizer = mobileclip.get_tokenizer('mobileclip_s1')
    text = tokenizer(KEYWORDS)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    app.logger.info("Models loaded successfully!")
    return model, preprocess, text_features

model, preprocess, text_features = load_models()
# Set up logging
logging.basicConfig(level=logging.DEBUG)


@app.route('/extract-images', methods=['POST'])
def extract_image():
    """Extracts keywords and embeddings from a batch of images uploaded in the request in parallel

    Returns:
        JSON: Embeddings and keywords in JSON format
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_files = request.files.getlist('image')
    print(f'Number of images in request: {len(image_files)}')

    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_image, image_files))

    print(f'Processing Complete!')
    return jsonify(results)


def process_image(image_file):
    """Processes a single image file and returns the embeddings and keywords of the image.

    Args:
        image_file (bytes): Image file uploaded in the request

    Returns:
        dict: Dictionary with embeddings and keywords of the single image file.
    """
    try:
        image = load_image(image_file)
        embeddings, keywords = inference(image)
        print(f"keywords: {keywords}")
        return {"embeddings": embeddings, "keywords": keywords}
    except Exception as e:
        return {"error": str(e)}

def inference(image):
    """Performs inference on the image provided.

    Args:
        image (PIL.Image.Image): Processed image file

    Returns:
        list (float) length 512: image embeddings
        list (str): list of keywords
    """
    keywords = []
    # Preprocess the image (resize, crop, normalize)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        # Extract image features and normalize
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute text probabilities based on image-text similarity
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        for idx, prob in enumerate(text_probs[0].tolist()):
            if prob > 0.50:
                keywords.append(KEYWORDS[idx])

    return image_features[0].cpu().tolist(), keywords

def load_image(image_file):
    """Converts uploaded image file bytes into PIL Image

    Args:
        image_file (bytes): Image file uploaded in the request

    Raises:
        ValueError: Image file is corrupted and unable to be decoded

    Returns:
        PIL.Image.Image : Image file is converted into PIL Image object.
    """
    image_file.seek(0)
    try:
        if image_file.filename.lower().endswith('.heic'):
            # Handle HEIC image files
            heif_image = pillow_heif.read_heif(image_file)
            return Image.frombytes(heif_image.mode, heif_image.size, heif_image.data).convert("RGB")
        else:
            # Handle other image formats using OpenCV
            file_bytes = image_file.read()
            image = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image.")
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
