# Load model directly
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
import time


start_time = time.time()
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k",use_fast=True)
model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
end_time = time.time()
print(f'Load model: {end_time-start_time:.2f}s')

start_time = time.time()
images = [f'./images/{file}' for file in os.listdir('./images')]
image_objects = []
no = 1
for _ in range(no):
    curr = images.pop(0)
    print(f'    image:{curr}')
    i = Image.open(curr)
    image_objects.append(i)
inputs = processor(images=image_objects, return_tensors="pt")
end_time = time.time()
print(f'Load images: {end_time-start_time:.2f}s')

start_time = time.time()
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
end_time = time.time()
print(f'Inference: {end_time-start_time:.2f}s\n\n')

print(last_hidden_states.shape)
print(f'Rows:{len(last_hidden_states[0])}')
print(f'Cols:{len(last_hidden_states[0][0])}')