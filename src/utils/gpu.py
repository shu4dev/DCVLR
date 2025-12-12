"""
GPU utilities for multi-GPU management and allocation.
"""

import logging
import torch

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU allocation for models."""

    def __init__(self):
        """Initialize GPU manager and detect available GPUs."""
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_info = []

        if self.num_gpus > 0:
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = self._get_gpu_memory(i)
                self.gpu_info.append({
                    'id': i,
                    'name': props.name,
                    'total_memory': total_mem,
                    'free_memory': free_mem,
                    'capability': f"{props.major}.{props.minor}"
                })

            logger.info(f"Detected {self.num_gpus} GPU(s)")
            for info in self.gpu_info:
                logger.info(
                    f"  GPU {info['id']}: {info['name']} - "
                    f"{info['free_memory']:.2f}GB free / {info['total_memory']:.2f}GB total"
                )
        else:
            logger.warning("No CUDA GPUs detected, will use CPU")

    def _get_gpu_memory(self, device_id: int):
        """
        Get free and total memory for a GPU device.

        Args:
            device_id: GPU device ID

        Returns:
            Tuple of (free_memory_gb, total_memory_gb)
        """
        torch.cuda.set_device(device_id)
        free = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        return free, total

    def get_device_for_model(self, model_name: str, prefer_gpu: int = None):
        """
        Get the best device for loading a model.

        Args:
            model_name: Name of the model (for logging)
            prefer_gpu: Preferred GPU ID, if None will auto-select

        Returns:
            Device string (e.g., 'cuda:0', 'cuda:1', or 'cpu')
        """
        if self.num_gpus == 0:
            logger.info(f"{model_name}: Using CPU (no GPUs available)")
            return "cpu"

        if prefer_gpu is not None and prefer_gpu < self.num_gpus:
            device = f"cuda:{prefer_gpu}"
            logger.info(f"{model_name}: Using {device} (user specified)")
            return device

        # Auto-select GPU with most free memory
        best_gpu = max(range(self.num_gpus),
                      key=lambda i: self._get_gpu_memory(i)[0])
        device = f"cuda:{best_gpu}"

        free_mem, _ = self._get_gpu_memory(best_gpu)
        logger.info(
            f"{model_name}: Auto-selected {device} "
            f"({free_mem:.2f}GB free)"
        )

        return device

    def get_model_distribution(self):
        """
        Get recommended model-to-GPU distribution based on available GPUs.

        Returns:
            Dictionary mapping model types to GPU IDs
        """
        if self.num_gpus == 0:
            return {
                'ocr': 'cpu',
                'yolo': 'cpu',
                'clip': 'cpu',
                'blip': 'cpu'
            }
        elif self.num_gpus == 1:
            # Single GPU: everything on GPU 0
            return {
                'ocr': 'cuda:0',
                'yolo': 'cuda:0',
                'clip': 'cuda:0',
                'blip': 'cuda:0'
            }
        elif self.num_gpus == 2:
            # Two GPUs: DeepSeek-OCR on GPU 0 (needs most memory)
            # Other models on GPU 1
            return {
                'ocr': 'cuda:0',
                'yolo': 'cuda:1',
                'clip': 'cuda:1',
                'blip': 'cuda:1'
            }
        else:
            # 3+ GPUs: spread models across GPUs
            return {
                'ocr': 'cuda:0',   # DeepSeek-OCR on GPU 0
                'yolo': 'cuda:1',  # YOLO on GPU 1
                'clip': 'cuda:2',  # CLIP on GPU 2
                'blip': 'cuda:2'   # BLIP on GPU 2 (shares with CLIP)
            }

    def clear_cache(self):
        """Clear CUDA cache on all GPUs."""
        if self.num_gpus > 0:
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache on all GPUs")

    def print_memory_summary(self):
        """Print memory usage summary for all GPUs."""
        if self.num_gpus == 0:
            logger.info("No GPUs available")
            return

        logger.info("=" * 60)
        logger.info("GPU Memory Summary:")
        logger.info("=" * 60)

        for i in range(self.num_gpus):
            free, total = self._get_gpu_memory(i)
            used = total - free
            usage_pct = (used / total) * 100 if total > 0 else 0

            logger.info(
                f"GPU {i}: {used:.2f}GB / {total:.2f}GB used ({usage_pct:.1f}%)"
            )
        logger.info("=" * 60)
