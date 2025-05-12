# fyp-models
This repository contains the code required to function the server component which hosts MobileCLIP, the AI model, for inferencing on images. This is used in conjunction with [SnapSage](https://github.com/marcusyks/image-search-app)

## Model used:
- `MobileCLIP` for both feature extraction and image to text
( note that numpy version has to be updated to "numpy<v2" )

## Idea:
- Use a `Flask` server to run MobileCLIP and extract feature and keywords via POST API call


## How to use:

1. Clone git repository, update submodule
    ```bash
    git clone git@github.com:marcusyks/fyp-models.git
    cd fyp-models
    git submodule update --init --recursive
    ```
2. Move **serverv2.py** and **requirements.txt** into submodule folder: mobileclip (replace requirements.txt in mobileclip folder)
    ```bash
    mv serverv2.py requirements.txt mobileclip
    ```

3. Create venv in mobileclip folder
    ```bash
    cd mobileclip
    python -m venv venv
    ```

4. Activate venv and install required dependencies
    ```bash
    source venv/bin/activate
    pip install -r requirements.txt
    ```

5. Download **mobileclip_s1.pt** for usage in serverv2.py (NOTE this will get all versions of mobileclip)
   ```bash
   source get_pretrained_models.sh 
   ```

6. Run Flask server
    ```bash
    python -m serverv2
    ```

7. Adjust frontend API call destination according to server IP

## References

This project utilizes MobileCLIP, a lightweight adaptation of OpenAI's CLIP model optimized for mobile and edge devices. MobileCLIP enables efficient image-text embedding extraction, making it suitable for on-device applications without requiring powerful GPUs.

For more details, refer to:
- [MobileCLIP Paper](https://arxiv.org/pdf/2311.17049)
- [Original CLIP by OpenAI](https://github.com/openai/CLIP)

