import torch
import time
import torch.nn as nn

def check_gpu():
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
    else:
        print("CUDA Available: NO (Using CPU)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock Model (similar to config)
    model = nn.Sequential(
        nn.Linear(384, 1024),
        nn.ReLU(),
        nn.Linear(1024, 384)
    ).to(device)

    batch_size = 8192
    input_tensor = torch.randn(batch_size, 384, device=device)

    print(f"\nBenchmarking Inference on {device} (Batch Size: {batch_size})...")

    # Warmup
    for _ in range(10):
        _ = model(input_tensor)

    start_time = time.time()
    iters = 100
    for _ in range(iters):
        _ = model(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    total_time = end_time - start_time
    total_items = iters * batch_size
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {total_items / total_time:.2f} items/sec")

if __name__ == "__main__":
    check_gpu()
