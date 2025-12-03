# Multi-GPU Optimization Guide

## Current Setup (With 2 GPUs)

The system already distributes models across GPUs:
- **GPU 0**: OCR (PaddleOCR or DeepSeek-OCR)
- **GPU 1**: YOLO + CLIP + BLIP

This provides **model-level parallelism** where different models run on different GPUs simultaneously.

## Current Bottleneck

Images are processed **sequentially** one at a time:
```python
for image in images:
    # All models run, but on next image only after current finishes
    text = ocr(image)      # GPU 0
    objects = yolo(image)  # GPU 1
    caption = blip(image)  # GPU 1
    similarity = clip(image, caption)  # GPU 1
```

## Optimization Strategies

### 1. Batch Processing (Easy Implementation)

**Speedup**: 2-3x for 100 images

Modify `bin_images()` to process in batches:

```python
def bin_images_batched(self, images: List[Dict], batch_size: int = 8):
    """Process images in batches for better GPU utilization."""
    bins = {'A': [], 'B': [], 'C': []}

    for batch_start in range(0, len(images), batch_size):
        batch = images[batch_start:batch_start + batch_size]

        # Process entire batch at once
        batch_paths = [img['path'] for img in batch]

        # Parallel execution across GPUs
        text_results = self.detect_text_batch(batch_paths)      # GPU 0
        object_results = self.detect_objects_batch(batch_paths) # GPU 1
        captions = self.generate_caption_batch(batch_paths)     # GPU 1
        similarities = self.calculate_clip_similarity_batch(    # GPU 1
            batch_paths, captions
        )

        # Categorize each image in batch
        for i, img_data in enumerate(batch):
            bin_cat = self._categorize_from_results(
                text_results[i], object_results[i],
                captions[i], similarities[i]
            )
            bins[bin_cat].append(img_data)

    return bins
```

**Pros**:
- Easy to implement
- No architecture changes needed
- Good speedup (2-3x)

**Cons**:
- Still processes batches sequentially
- GPU idle time between batches

### 2. Multi-Process Parallelism (Best for 2 GPUs)

**Speedup**: 1.5-1.9x (near-linear scaling)

Split images across 2 processes, each using one GPU:

```python
import torch.multiprocessing as mp

def process_images_on_gpu(gpu_id, images, config, result_queue):
    """Process images on a specific GPU."""
    # Set this process to use only the assigned GPU
    torch.cuda.set_device(gpu_id)

    # Initialize binner with single GPU
    binner_config = config.copy()
    binner_config['enable_multi_gpu'] = False
    binner = ImageBinner(binner_config)

    # Force models to this GPU
    # ... (model initialization code)

    # Process images
    bins = binner.bin_images(images)
    result_queue.put((gpu_id, bins))

def bin_images_multi_gpu(self, images: List[Dict]):
    """Distribute image binning across multiple GPUs."""
    n_gpus = torch.cuda.device_count()

    if n_gpus < 2:
        # Fall back to single GPU
        return self.bin_images(images)

    # Split images evenly
    chunk_size = len(images) // n_gpus
    image_chunks = [
        images[i*chunk_size:(i+1)*chunk_size]
        for i in range(n_gpus)
    ]

    # Create processes
    result_queue = mp.Queue()
    processes = []

    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=process_images_on_gpu,
            args=(gpu_id, image_chunks[gpu_id], self.config, result_queue)
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

    # Merge results
    bins = {'A': [], 'B': [], 'C': []}
    for _ in range(n_gpus):
        gpu_id, gpu_bins = result_queue.get()
        for bin_key in bins:
            bins[bin_key].extend(gpu_bins[bin_key])

    return bins
```

**Pros**:
- Near-linear speedup (1.8-2x with 2 GPUs)
- Both GPUs work independently
- Simple concept

**Cons**:
- Requires multiprocessing setup
- Memory overhead (each process loads models)
- More complex error handling

### 3. Async Pipeline (Advanced)

**Speedup**: 3-5x with optimal overlap

Create an async pipeline with queues:

```python
import asyncio
from queue import Queue
from threading import Thread

class AsyncImageBinner:
    def __init__(self, config):
        self.config = config
        self.ocr_queue = Queue(maxsize=10)
        self.detection_queue = Queue(maxsize=10)
        self.results_queue = Queue()

    def ocr_worker(self):
        """Worker for OCR on GPU 0."""
        while True:
            image_data = self.ocr_queue.get()
            if image_data is None:
                break
            text_result = self.detect_text(image_data['path'])
            image_data['text_result'] = text_result
            self.detection_queue.put(image_data)

    def detection_worker(self):
        """Worker for detection/captioning on GPU 1."""
        while True:
            image_data = self.detection_queue.get()
            if image_data is None:
                break
            # Run all GPU 1 tasks
            image_data['object_result'] = self.detect_objects(image_data['path'])
            image_data['caption'] = self.generate_caption(image_data['path'])
            image_data['similarity'] = self.calculate_clip_similarity(
                image_data['path'], image_data['caption']
            )
            self.results_queue.put(image_data)

    def bin_images_async(self, images):
        """Process images through async pipeline."""
        # Start workers
        ocr_thread = Thread(target=self.ocr_worker)
        detection_thread = Thread(target=self.detection_worker)
        ocr_thread.start()
        detection_thread.start()

        # Feed images
        for img in images:
            self.ocr_queue.put(img)

        # Signal completion
        self.ocr_queue.put(None)

        # Collect results
        bins = {'A': [], 'B': [], 'C': []}
        for _ in range(len(images)):
            result = self.results_queue.get()
            bin_cat = self._categorize(result)
            bins[bin_cat].append(result)

        # Cleanup
        self.detection_queue.put(None)
        ocr_thread.join()
        detection_thread.join()

        return bins
```

**Pros**:
- Maximum GPU utilization
- Best speedup (3-5x)
- Smooth pipeline flow

**Cons**:
- Complex implementation
- Requires careful queue management
- Debugging is harder

## Recommended Approach

### For Quick Wins (Today):
**Use Batch Processing** - Modify existing batch methods to process multiple images at once:
- `detect_text_batch()`
- `detect_objects_batch()`
- `generate_caption_batch()` (already exists!)
- `calculate_clip_similarity_batch()`

### For Maximum Performance (This Week):
**Use Multi-Process Parallelism** - Split images across both GPUs:
- Each GPU gets 50% of images
- Both run completely independently
- Merge results at the end

### Performance Estimates (100 images):

| Method | Time | Speedup |
|--------|------|---------|
| Current (Sequential) | 35s | 1x |
| Batch Processing (size=8) | 15s | 2.3x |
| Multi-Process (2 GPUs) | 18s | 1.9x |
| Async Pipeline | 10s | 3.5x |

## Implementation Priority

1. ✅ **Already done**: Model distribution across GPUs
2. 🎯 **Easy win**: Batch processing (add `batch_size` parameter)
3. 🎯 **Best ROI**: Multi-process parallelism
4. 🔬 **Advanced**: Async pipeline (if you need max speed)

## Code Locations to Modify

- [src/filtering/binning.py:1131-1166](src/filtering/binning.py#L1131-L1166) - Main processing loop
- [src/filtering/binning.py:454-579](src/filtering/binning.py#L454-L579) - `detect_text()`
- [src/filtering/binning.py:644-701](src/filtering/binning.py#L644-L701) - `detect_objects_yolo()`
- [src/filtering/binning.py:1238-1266](src/filtering/binning.py#L1238-L1266) - `generate_captions_batch()` (already batched!)

## Next Steps

Would you like me to implement any of these optimizations? I recommend starting with **Option 2 (Multi-Process)** as it provides the best balance of speedup and implementation complexity for 2 GPUs.
