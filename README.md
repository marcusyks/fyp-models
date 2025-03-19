# fyp-models

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

5. Download **mobileclip_s1.pt** for usage in serverv2.py
   ```bash
   huggingface-cli download pcuenq/MobileCLIP-S1
   ```

6. Run Flask server
    ```bash
    python -m serverv2
    ```

7. Adjust frontend API call destination according to server IP


