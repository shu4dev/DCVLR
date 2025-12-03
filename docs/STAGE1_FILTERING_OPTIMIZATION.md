# Stage 1 Filtering Optimization Guide

## Current Implementation Analysis

Looking at [image_filter.py](../src/filtering/image_filter.py) and [team1_pipeline.py:282-290](../team1_pipeline.py#L282-L290), Stage 1 filtering is currently **sequential and single-threaded**:

```python
for img_path in images:
    if check_resolution(img_path):      # CPU - Fast (~1ms)
        if check_nsfw(img_path):        # GPU - Slow (~50-100ms)
            if not is_duplicate(img_path):  # CPU - Fast (~5ms)
                filtered_images.append(img_path)
```

## Performance Bottleneck: NSFW Detection

### Time Breakdown (per image):

| Operation | Time | Device | Parallelizable? |
|-----------|------|--------|-----------------|
| Resolution check | ~1ms | CPU | ✅ Yes |
| **NSFW detection** | **~50-100ms** | **GPU** | ✅ **Yes** |
| Duplicate check | ~5ms | CPU | ⚠️ Limited |
| Total | ~56-106ms | - | - |

**NSFW detection takes 90%+ of the time!**

## Optimization Opportunities

### 1. **Batch NSFW Detection** ⭐ (Best ROI)

**Impact**: 5-10x speedup for filtering stage
**Complexity**: Low
**Memory**: Moderate

Instead of checking one image at a time, batch multiple images together:

```python
# Current (Sequential): ~100ms per image
for img in images:
    result = nsfw_detector(img)  # GPU underutilized

# Optimized (Batched): ~20ms per image
batch_size = 16
for batch in batched(images, batch_size):
    results = nsfw_detector(batch)  # GPU fully utilized
```

**Performance**:
- Single GPU: **5-8x faster**
- 100 images: 10s → **1.5s** ⚡

### 2. **Early Filtering** (Free optimization)

Check cheap filters first to avoid expensive GPU calls:

```python
# Current order: Resolution → NSFW → Duplicate
# Problem: Still runs NSFW on images that will fail duplicate check

# Optimized order: Resolution → Duplicate → NSFW
for img_path in images:
    if check_resolution(img_path):     # Fast CPU check
        if not is_duplicate(img_path):  # Fast CPU check
            if check_nsfw(img_path):    # Expensive GPU check
                filtered_images.append(img_path)
```

**Benefit**: Skip GPU calls for duplicates (~10-20% reduction)

### 3. **Multi-GPU Filtering** (When you have 2+ GPUs)

**Impact**: 1.8-1.9x speedup
**Complexity**: Medium
**Memory**: High (each GPU loads NSFW model)

Split images across GPUs with data parallelism:

```python
# GPU 0: Check images 0-49 (NSFW model on GPU 0)
# GPU 1: Check images 50-99 (NSFW model on GPU 1)
```

**Performance**:
- 2 GPUs: **~1.9x faster**
- 4 GPUs: **~3.7x faster**

### 4. **Async I/O + GPU Pipeline**

**Impact**: 1.5-2x speedup
**Complexity**: High

Create pipeline where I/O and GPU work in parallel:

```
Thread 1 (I/O): Load batch N+1 from disk
Thread 2 (GPU): Process batch N with NSFW detection
```

**Performance**: ~1.5-2x faster, best combined with batching

### 5. **Caching** (One-time cost savings)

Cache filter results if you process same images multiple times:

```python
# Save results to disk
cache_file = "filter_cache.json"
cache = {
    'image_path': {'resolution': True, 'nsfw': False, 'duplicate': False}
}
```

**Benefit**: Skip re-processing, useful for development/testing

## Recommended Implementation

### Priority 1: Batch NSFW Detection (Implement First)

Modify `ImageFilter.check_nsfw()` to support batching:

```python
def check_nsfw_batch(self, image_paths: List[str], batch_size: int = 16) -> List[bool]:
    """
    Check multiple images for NSFW content in batches.

    Args:
        image_paths: List of image paths
        batch_size: Number of images to process at once

    Returns:
        List of boolean values (True = safe, False = NSFW)
    """
    if not self.nsfw_detector:
        return [True] * len(image_paths)

    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        # Load batch
        for path in batch_paths:
            try:
                img = Image.open(path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                batch_images.append(img)
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
                results.append(False)
                continue

        # Process entire batch at once
        if batch_images:
            predictions = self.nsfw_detector(batch_images)

            # Parse results
            for pred in predictions:
                is_safe = True
                for result in pred:
                    if result['label'].lower() == 'nsfw' and result['score'] > self.nsfw_threshold:
                        is_safe = False
                        break
                results.append(is_safe)

    return results
```

### Priority 2: Reorder Filters

Update [team1_pipeline.py:282-290](../team1_pipeline.py#L282-L290):

```python
# Phase 1: Fast CPU filters
cpu_filtered = []
for img_path in tqdm(images_to_check, desc="CPU Filtering"):
    if self.image_filter.check_resolution(img_path):
        if not self.image_filter.is_duplicate(img_path):
            cpu_filtered.append(img_path)

logger.info(f"CPU filters: {len(images_to_check)} → {len(cpu_filtered)} images")

# Phase 2: Batch GPU filtering
batch_size = 16
filtered_images = []

for i in range(0, len(cpu_filtered), batch_size):
    batch = cpu_filtered[i:i+batch_size]
    nsfw_results = self.image_filter.check_nsfw_batch(batch, batch_size)

    for path, is_safe in zip(batch, nsfw_results):
        if is_safe:
            filtered_images.append({
                'path': path,
                'id': Path(path).stem
            })

logger.info(f"GPU filters: {len(cpu_filtered)} → {len(filtered_images)} images")
```

### Priority 3: Multi-GPU (If you have 2+ GPUs)

Similar to binning multiprocess, split filtering across GPUs:

```python
class MultiGPUImageFilter:
    def filter_images_parallel(self, images, num_gpus=2):
        # Split images across GPUs
        chunks = split_evenly(images, num_gpus)

        # Process in parallel
        with mp.Pool(num_gpus) as pool:
            results = pool.starmap(
                filter_on_gpu,
                [(chunk, gpu_id) for gpu_id, chunk in enumerate(chunks)]
            )

        # Merge results
        return flatten(results)
```

## Performance Estimates

### Current Performance (100 images):
- Resolution check: 0.1s (100 × 1ms)
- NSFW detection: **10s** (100 × 100ms) ← **Bottleneck**
- Duplicate check: 0.5s (100 × 5ms)
- **Total: ~10.6s**

### With Batch Processing (Priority 1 + 2):
- CPU filters: 0.6s
- NSFW batch (batch=16): **1.5s** (7× faster)
- **Total: ~2.1s** ⚡ **(5x speedup)**

### With Multi-GPU (2 GPUs):
- Each GPU processes 50 images
- Time per GPU: ~1.0s
- **Total: ~1.0s** ⚡⚡ **(10x speedup)**

### With All Optimizations:
- CPU filters + Batching + 2 GPUs + Async I/O
- **Total: ~0.7s** ⚡⚡⚡ **(15x speedup)**

## Implementation Complexity vs Benefit

| Optimization | Speedup | Complexity | Memory | Priority |
|--------------|---------|------------|--------|----------|
| **Batch NSFW** | **5-8x** | **Low** | **Medium** | **🥇 1** |
| Reorder filters | 1.2-1.5x | Very Low | None | 🥈 2 |
| Multi-GPU | 1.9x | Medium | High | 🥉 3 |
| Async I/O | 1.5x | High | Low | 4 |
| Caching | ∞ (reuse) | Low | Low | 5 |

## Quick Win: Batching Implementation

Here's a drop-in replacement you can use immediately:

```python
# In image_filter.py, add this method to ImageFilter class:

def filter_images_batched(self, image_paths: List[str], batch_size: int = 16) -> List[str]:
    """
    Apply all filters with batched GPU processing for speed.

    Args:
        image_paths: List of paths to image files
        batch_size: Batch size for GPU operations (default: 16)

    Returns:
        List of paths that passed all filters
    """
    # Phase 1: Fast CPU filters (resolution + duplicate)
    cpu_passed = []
    for path in image_paths:
        if self.check_resolution(path) and not self.is_duplicate(path):
            cpu_passed.append(path)

    logger.info(f"CPU filters: {len(image_paths)} → {len(cpu_passed)} images")

    # Phase 2: Batched GPU filter (NSFW)
    filtered = []
    for i in range(0, len(cpu_passed), batch_size):
        batch = cpu_passed[i:i+batch_size]
        nsfw_results = self.check_nsfw_batch(batch, batch_size)

        for path, is_safe in zip(batch, nsfw_results):
            if is_safe:
                filtered.append(path)

    logger.info(f"Total filtered: {len(image_paths)} → {len(filtered)} images")
    return filtered
```

Then in `team1_pipeline.py`, replace the loop with:

```python
# Get list of paths
paths = [img['path'] for img in images_to_check]

# Filter in batches
passed_paths = self.image_filter.filter_images_batched(paths, batch_size=16)

# Convert back to dict format
filtered_images = [
    {'path': path, 'id': Path(path).stem}
    for path in passed_paths
]
```

## Comparison: Stage 1 vs Stage 2 Optimization

| Metric | Stage 1 (Filtering) | Stage 2 (Binning) |
|--------|---------------------|-------------------|
| **Current bottleneck** | NSFW (GPU) | All models |
| **Best optimization** | Batching | Multi-GPU |
| **Quick win speedup** | 5-8x | 1.9x |
| **Max speedup** | 15x | 3.7x (4 GPUs) |
| **Implementation difficulty** | Low | Medium |

**Stage 1 has MORE optimization potential than Stage 2!**

## Next Steps

1. **Immediate** (Today): Implement batch NSFW detection
   - Add `check_nsfw_batch()` method
   - Modify pipeline to use batching
   - Expected: 5x speedup

2. **Short-term** (This week): Reorder filters
   - Move duplicate check before NSFW
   - Expected: +20% improvement

3. **Medium-term** (Next week): Multi-GPU filtering
   - Implement parallel filtering across GPUs
   - Expected: +90% improvement (2x)

4. **Long-term**: Async I/O pipeline
   - For maximum performance
   - Expected: +50% improvement

## Summary

**Stage 1 filtering has HUGE optimization potential!**

- **Current**: ~10.6s for 100 images (bottleneck: NSFW detection)
- **With batching**: ~2.1s **(5x faster)** ← Easy to implement!
- **With multi-GPU**: ~1.0s **(10x faster)** ← Best for 2+ GPUs
- **With all optimizations**: ~0.7s **(15x faster)** ← Maximum performance

**Recommendation**: Start with batch NSFW detection - it's the easiest and gives the biggest speedup!
