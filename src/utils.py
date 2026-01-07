import torch

def get_device() -> torch.device:
    """
    Returns the best available device:
    1. CUDA (NVIDIA)
    2. MPS (Apple Silicon)
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
