import os
import requests

MODEL_URL = "https://huggingface.co/USERNAME/fakenews-model/resolve/main/model.pkl"
VOCAB_URL = "https://huggingface.co/USERNAME/fakenews-model/resolve/main/vocab.pkl"

def download(url, filename):
    if os.path.exists(filename):
        return
    r = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        for chunk in r.iter_content(1024*1024):
            f.write(chunk)

def download_artifacts():
    download(MODEL_URL, "model.pkl")
    download(VOCAB_URL, "vocab.pkl")
