# Multi-GPU Setup Guide

This guide explains how the pipeline automatically distributes models across multiple GPUs to overcome memory limitations.

## Problem

DeepSeek-OCR (even the 'tiny' model) requires approximately **10+ GiB** of VRAM, which nearly fills a single RTX 2080 Ti (10.57 GiB). When combined with other models (YOLO, CLIP, BLIP), this causes out-of-memory errors.

## Solution: Automatic Multi-GPU Distribution

The pipeline now **automatically detects available GPUs** and distributes models optimally:

### Single GPU (e.g., 1x RTX 2080 Ti)
- **Status**: Will likely fail with OOM errors
- **Distribution**: All models on `cuda:0`
- **Recommendation**: Use 2+ GPUs or switch to CPU for some models

### Two GPUs (e.g., 2x RTX 2080 Ti) ✅ Recommended
- **DeepSeek-OCR**: `cuda:0` (~10 GiB)
- **YOLO**: `cuda:1` (~0.5 GiB)
- **CLIP**: `cuda:1` (~0.5 GiB)
- **BLIP**: `cuda:1` (~2 GiB)
- **Total**: GPU0: ~10GB, GPU1: ~3GB
- **Status**: Should work well!

### Three+ GPUs (e.g., 3x RTX 2080 Ti)
- **DeepSeek-OCR**: `cuda:0` (~10 GiB)
- **YOLO**: `cuda:1` (~0.5 GiB)
- **CLIP**: `cuda:2` (~0.5 GiB)
- **BLIP**: `cuda:2` (~2 GiB)
- **Status**: Optimal distribution

## Configuration

The multi-GPU feature is **enabled by default** in [configs/default_config.yaml](../configs/default_config.yaml):

```yaml
binning:
  enable_multi_gpu: true  # Auto-detect and use all available GPUs
```

To disable multi-GPU (force single device):

```yaml
binning:
  enable_multi_gpu: false  # All models will use cuda:0 or CPU
```

## How It Works

1. **Automatic Detection**: The `GPUManager` class detects all available GPUs at startup
2. **Smart Distribution**: Models are distributed based on the number of GPUs:
   - Priority: DeepSeek-OCR always gets its own GPU (largest model)
   - Other models share remaining GPUs
3. **Memory Reporting**: Displays memory usage for each GPU during initialization

## Verifying Multi-GPU Usage

When you run the pipeline, you'll see logs like:

```
INFO - Detected 2 GPU(s)
INFO - GPU 0: NVIDIA GeForce RTX 2080 Ti - 10.50GB free / 10.57GB total
INFO - GPU 1: NVIDIA GeForce RTX 2080 Ti - 10.50GB free / 10.57GB total
INFO - Multi-GPU enabled with 2 GPU(s)
INFO - Model distribution: {'ocr': 'cuda:0', 'yolo': 'cuda:1', 'clip': 'cuda:1', 'blip': 'cuda:1'}
INFO - Loading DeepSeek-OCR on cuda:0...
INFO - DeepSeek-OCR loaded successfully on cuda:0
INFO - Loading YOLO on cuda:1...
INFO - YOLO loaded on cuda:1
...
```

## Model Size Options

You can also adjust model sizes in the config:

```yaml
binning:
  deepseek_model_size: 'tiny'   # Options: 'tiny', 'small', 'base', 'large'
  use_blip2: false              # true = BLIP-2 (better quality, more memory)
```

### DeepSeek-OCR Model Sizes (Approximate)
- **tiny**: ~10 GiB VRAM (base_size=512)
- **small**: ~15 GiB VRAM (estimate)
- **base**: ~20 GiB VRAM (requires 2 GPUs)
- **large**: ~30+ GiB VRAM (requires 3+ GPUs)

## Monitoring GPU Usage

Check GPU usage during runtime:

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

You should see:
- GPU 0: High memory usage (~10 GB from DeepSeek-OCR)
- GPU 1: Lower memory usage (~3 GB from YOLO+CLIP+BLIP)

## Troubleshooting

### Still getting OOM errors with 2 GPUs?

1. **Check GPU visibility**:
   ```bash
   python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
   ```

2. **Clear GPU cache before running**:
   ```bash
   # Kill any existing Python processes using GPUs
   pkill -9 python

   # Run your pipeline
   python pipeline_demo.py
   ```

3. **Try smaller model**:
   - DeepSeek-OCR is already at 'tiny'
   - Ensure `use_blip2: false` (BLIP-2 needs more memory)

### Want to use larger models?

With 2x RTX 2080 Ti, you can try:
```yaml
binning:
  deepseek_model_size: 'small'  # ~15 GiB (might work on 2 GPUs)
  use_blip2: true               # Better captions (adds ~3 GiB)
```

With 3+ GPUs, you can use:
```yaml
binning:
  deepseek_model_size: 'base'   # ~20 GiB (needs dedicated GPU)
  use_blip2: true               # ~3 GiB (on separate GPU)
```

## Files Modified

- [src/utils/gpu_utils.py](../src/utils/gpu_utils.py) - GPU detection and management
- [src/filtering/binning.py](../src/filtering/binning.py) - Model initialization with multi-GPU support
- [configs/default_config.yaml](../configs/default_config.yaml) - Configuration options

## Summary

✅ **With 2x RTX 2080 Ti**: Your pipeline should now work!
- GPU 0: DeepSeek-OCR (~10 GB)
- GPU 1: YOLO + CLIP + BLIP (~3 GB)

✅ **Automatic**: No manual configuration needed
✅ **Scalable**: Works with 1, 2, 3+ GPUs
✅ **Transparent**: Shows GPU allocation in logs
