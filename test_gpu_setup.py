#!/usr/bin/env python3
"""
Quick test script to verify GPU detection and allocation.
Run this before running the full pipeline to ensure multi-GPU setup is working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.gpu_utils import GPUManager


def main():
    print("=" * 70)
    print("GPU Detection and Allocation Test")
    print("=" * 70)

    # Initialize GPU manager
    gpu_manager = GPUManager()

    print(f"\n📊 Number of GPUs detected: {gpu_manager.num_gpus}")

    if gpu_manager.num_gpus == 0:
        print("\n⚠️  WARNING: No CUDA GPUs detected!")
        print("   The pipeline will run on CPU (very slow)")
        print("   Make sure CUDA is installed and GPUs are available")
        return

    print("\n💾 GPU Information:")
    for info in gpu_manager.gpu_info:
        print(f"   GPU {info['id']}: {info['name']}")
        print(f"     Total Memory: {info['total_memory']:.2f} GB")
        print(f"     Free Memory:  {info['free_memory']:.2f} GB")
        print(f"     Compute Capability: {info['capability']}")
        print()

    # Show recommended model distribution
    print("🎯 Recommended Model Distribution:")
    distribution = gpu_manager.get_model_distribution()
    for model, device in distribution.items():
        print(f"   {model.upper():15s} -> {device}")

    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)

    if gpu_manager.num_gpus == 1:
        print("⚠️  Single GPU detected:")
        print("   - All models will load on cuda:0")
        print("   - DeepSeek-OCR alone needs ~10 GB")
        print("   - You may encounter out-of-memory errors")
        print("   - RECOMMENDATION: Use 2+ GPUs if possible")
    elif gpu_manager.num_gpus == 2:
        print("✅ Two GPUs detected - OPTIMAL for this pipeline!")
        print("   - DeepSeek-OCR on GPU 0 (~10 GB)")
        print("   - YOLO + CLIP + BLIP on GPU 1 (~3 GB)")
        print("   - This should work without memory issues")
    else:
        print(f"✅ {gpu_manager.num_gpus} GPUs detected - EXCELLENT!")
        print("   - Models will be distributed across all GPUs")
        print("   - Optimal memory usage and performance")

    # Memory summary
    print("\n" + "=" * 70)
    gpu_manager.print_memory_summary()

    print("\n✅ GPU test complete!")
    print("   You can now run: python pipeline_demo.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
