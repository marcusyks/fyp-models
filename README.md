# fyp-models

## Models used:
- `Tag2Text` for keywords extraction
- `ViT` for image feature extraction

## Idea:
- `BLIP` to extract keywords from each image in gallery
- `ViT` to extract each image feature and provide similarity search

- Use a `Flask` server to run both models and extract feature and keywords via API call

## Changes:
- `Tag2Text` model to `BLIP` model: Computation time too long and model too big (40s/image - 30s/image) (60s/setup - 20s/setup)
- `flask[async]`: concurrent image extracting and keyword extracting (30s/image - 25s/image)
- ONNX conversion: make inference faster ()


