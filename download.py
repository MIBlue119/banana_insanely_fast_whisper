# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
import os
import torch
from transformers import (
    WhisperForConditionalGeneration,
)

def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise

def download_model(model_id, device, torch_dtype):
    model = fetch_pretrained_model(
        WhisperForConditionalGeneration,
        model_id,
        torch_dtype=torch_dtype
    ).to(device)

if __name__ == "__main__":
    if os.environ.get("HF_HOME") != "/cache/huggingface":
        print(f"HF_HOME is set to {os.environ.get('HF_HOME')}")
        raise ValueError("HF_HOME must be set to /cache/huggingface")
    download_model("openai/whisper-large-v3",
              "cuda:0" if torch.cuda.is_available() else "cpu", torch.float16)    