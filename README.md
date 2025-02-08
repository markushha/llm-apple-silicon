#### This repo is made for research purposes only.
### Local environment setup to run open-sourced LLMs (in this case KazLLM-1.0-8B) on macOS (Apple Silicon CPU) with token-by-token generation

### After cloning the repo
```
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
```

### download packages
```
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
```

### run the model
```
python run.py
```

Take a note that the model will run extremely slow on Apple Silicon CPU as it is uses CPU for inference.

### References
- [KazLLM-8B](https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B)
