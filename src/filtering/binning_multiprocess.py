"""
Multi-process data parallelism for image binning.
When GPU memory is sufficient, this provides near-linear speedup across GPUs.
"""

import logging
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def worker_process(
    gpu_id: int,
    image_chunk: List[Dict],
    config: Dict,
    return_dict: Dict,
    worker_id: int
):
    """
    Worker process that bins images on a specific GPU.

    Each worker:
    1. Sets its GPU device
    2. Loads all models (OCR, YOLO, CLIP, BLIP) on that GPU
    3. Processes its chunk of images
    4. Returns results

    Args:
        gpu_id: GPU device ID to use (0, 1, 2, etc.)
        image_chunk: List of images to process
        config: Configuration dictionary
        return_dict: Shared dictionary for results
        worker_id: Worker identifier
    """
    try:
        # Set this process to use only the assigned GPU
        torch.cuda.set_device(gpu_id)

        # Import here to avoid issues with multiprocessing
        from src.filtering.binning import ImageBinner

        logger.info(f"Worker {worker_id} starting on GPU {gpu_id} with {len(image_chunk)} images")

        # Force single-GPU mode for this worker
        worker_config = config.copy()
        worker_config['enable_multi_gpu'] = False

        # Initialize binner - all models will load on the assigned GPU
        binner = ImageBinner(worker_config)

        # Override device settings to force everything to this GPU
        device = f"cuda:{gpu_id}"
        binner.ocr_device = device
        binner.detector_device = device
        binner.clip_device = device
        if hasattr(binner, 'blip_device') and binner.blip_device:
            binner.blip_device = device

        logger.info(f"Worker {worker_id}: All models loaded on {device}")
        logger.info(f"Worker {worker_id}: Processing {len(image_chunk)} images...")

        # Process images
        bins = binner.bin_images(image_chunk)

        logger.info(f"Worker {worker_id} completed: A={len(bins['A'])}, B={len(bins['B'])}, C={len(bins['C'])}")

        # Store results
        return_dict[worker_id] = bins

    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {e}", exc_info=True)
        return_dict[worker_id] = {'A': [], 'B': [], 'C': [], 'error': str(e)}


class MultiProcessImageBinner:
    """
    Image binner that uses multiple GPUs in parallel via multiprocessing.

    This achieves near-linear speedup when GPU memory is sufficient to hold
    all models (OCR + YOLO + CLIP + BLIP) on each GPU.

    Example:
        With 2 GPUs and 100 images:
        - GPU 0 processes images 0-49
        - GPU 1 processes images 50-99
        Both run simultaneously → ~2x speedup
    """

    def __init__(self, config: Dict):
        """
        Initialize multi-process binner.

        Args:
            config: Configuration dictionary (same as ImageBinner)
        """
        self.config = config
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available for multi-process binning")

        logger.info(f"MultiProcessImageBinner initialized with {self.num_gpus} GPUs")

        # Check if multi-GPU is enabled
        if not config.get('enable_multi_gpu', True):
            logger.warning("enable_multi_gpu is False but using MultiProcessImageBinner")

    def bin_images(
        self,
        images: List[Dict],
        display_details: bool = False,
        user_criteria: Dict = None,
        dataset_size: Optional[int] = None,
        bin_ratios: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ) -> Dict[str, List[Dict]]:
        """
        Bin images using multiple GPUs in parallel.

        Args:
            images: List of image dictionaries with 'path' key
            display_details: If True, display detailed results (only works with single GPU)
            user_criteria: Optional user-defined criteria (not used in multi-process mode)
            dataset_size: Target total dataset size (if None, uses all images)
            bin_ratios: Tuple of (bin_a_ratio, bin_b_ratio, bin_c_ratio) for distribution

        Returns:
            Dictionary with bin categories as keys and image lists as values
        """
        # If only 1 GPU or very few images, fall back to single-process
        if self.num_gpus == 1 or len(images) < self.num_gpus * 4:
            logger.info("Using single-process mode (insufficient parallelism benefit)")
            from src.filtering.binning import ImageBinner
            binner = ImageBinner(self.config)
            return binner.bin_images(images, display_details, user_criteria, dataset_size, bin_ratios)

        logger.info(f"Starting multi-process binning:")
        logger.info(f"  Total images: {len(images)}")
        logger.info(f"  GPUs available: {self.num_gpus}")
        logger.info(f"  Images per GPU: ~{len(images) // self.num_gpus}")

        # Split images evenly across GPUs
        chunk_size = (len(images) + self.num_gpus - 1) // self.num_gpus  # Ceiling division
        image_chunks = [
            images[i:i + chunk_size]
            for i in range(0, len(images), chunk_size)
        ]

        # Ensure we don't have more chunks than GPUs
        image_chunks = image_chunks[:self.num_gpus]

        logger.info(f"Split into {len(image_chunks)} chunks:")
        for i, chunk in enumerate(image_chunks):
            logger.info(f"  GPU {i}: {len(chunk)} images")

        # Create shared dictionary for results (using Manager for process-safe sharing)
        manager = mp.Manager()
        return_dict = manager.dict()

        # Create and start worker processes
        processes = []
        for gpu_id, chunk in enumerate(image_chunks):
            if len(chunk) == 0:
                continue

            p = mp.Process(
                target=worker_process,
                args=(gpu_id, chunk, self.config, return_dict, gpu_id)
            )
            p.start()
            processes.append(p)

        logger.info(f"Started {len(processes)} worker processes")

        # Wait for all processes to complete
        for i, p in enumerate(processes):
            p.join()
            logger.info(f"Worker {i} finished")

        # Check for errors
        errors = []
        for worker_id in range(len(image_chunks)):
            if worker_id in return_dict and 'error' in return_dict[worker_id]:
                errors.append(f"Worker {worker_id}: {return_dict[worker_id]['error']}")

        if errors:
            logger.error(f"Errors in workers: {errors}")

        # Merge results from all workers
        logger.info("Merging results from all workers...")
        merged_bins = {'A': [], 'B': [], 'C': []}

        for worker_id in sorted(return_dict.keys()):
            worker_bins = return_dict[worker_id]
            if 'error' not in worker_bins:
                for bin_key in ['A', 'B', 'C']:
                    merged_bins[bin_key].extend(worker_bins[bin_key])

        # Log final results
        logger.info(f"Multi-process binning complete:")
        logger.info(f"  Bin A (Text/Arithmetic): {len(merged_bins['A'])} images")
        logger.info(f"  Bin B (Object/Spatial): {len(merged_bins['B'])} images")
        logger.info(f"  Bin C (Commonsense/Attribute): {len(merged_bins['C'])} images")
        logger.info(f"  Total binned: {sum(len(v) for v in merged_bins.values())} images")

        return merged_bins


def enable_multiprocess_binning():
    """
    Enable multiprocessing for the main process.
    Call this at the start of your script before creating the pipeline.
    """
    mp.set_start_method('spawn', force=True)
    logger.info("Multiprocessing enabled with 'spawn' method")
