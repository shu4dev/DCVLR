import time

try:
    import torch
except ImportError:
    raise SystemExit("You need to `pip install torch` first.")

def keep_gpu_running(matrix_size=1024, sleep_seconds=0.1):
    """
    Continuously runs small ops on the GPU to keep it active.

    matrix_size:    size of the square matrix (larger => more GPU load)
    sleep_seconds:  pause between iterations (smaller => more GPU load)
    """
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. Exiting.")

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Create some data on GPU
    x = torch.randn((matrix_size, matrix_size), device=device)
    w = torch.randn((matrix_size, matrix_size), device=device)

    print("Starting GPU keep-alive loop. Press Ctrl+C to stop.")
    try:
        while True:
            # Dummy workload (matrix multiply + nonlinearity)
            x = torch.sin(x @ w)

            # Make sure the operations are actually executed
            torch.cuda.synchronize()

            # Short sleep so it doesn’t peg the GPU at 100% constantly
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # Optional: free memory
        del x, w
        torch.cuda.empty_cache()
        print("Cleaned up GPU tensors.")


if __name__ == "__main__":
    # tweak these if you want more/less load
    keep_gpu_running(matrix_size=1024, sleep_seconds=0.1)
