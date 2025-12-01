"""
Image binning module for categorizing images into Text/Object/Commonsense bins.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import open_clip
try:
    from deep_ocr import DeepSeekOCR, OCRConfig
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logger.warning("DeepSeek-OCR not available, will use PaddleOCR")

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
import pandas as pd
import time
import re
import cv2

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.gpu_utils import GPUManager

logger = logging.getLogger(__name__)


class ImageBinner:
    """
    Categorizes images into three bins based on content:
    - Bin A: Text/Arithmetic (text-heavy images)
    - Bin B: Object/Spatial (object-rich images) 
    - Bin C: Commonsense/Attribute (general images)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the image binner with models and configuration.

        Args:
            config: Dictionary containing binning configuration
        """
        self.config = config

        # Thresholds from config
        self.text_boxes_threshold = config.get('text_boxes_threshold', 2)
        self.text_area_threshold = config.get('text_area_threshold', 0.2)
        self.object_count_threshold = config.get('object_count_threshold', 5)
        self.unique_objects_threshold = config.get('unique_objects_threshold', 3)
        self.clip_threshold = config.get('clip_similarity_threshold', 0.25)
        self.spatial_dispersion_threshold = config.get('spatial_dispersion_threshold', 0.3)

        # Model configuration
        self.use_blip2 = config.get('use_blip2', False)

        # Object detector selection
        self.object_detector = config.get('object_detector', 'yolo').lower()  # 'yolo' or 'sam'

        # YOLO-specific configuration
        self.yolo_model = config.get('yolo_model', 'yolov8n')

        # SAM-specific configuration
        self.sam_model_type = config.get('sam_model_type', 'vit_b')
        self.sam_checkpoint = config.get('sam_checkpoint', 'models/sam_vit_b_01ec64.pth')

        # GPU configuration
        self.enable_multi_gpu = config.get('enable_multi_gpu', True)  # Auto-detect and use multiple GPUs
        self.gpu_manager = GPUManager()

        # OCR backend selection
        self.use_paddle_ocr = config.get('use_paddle_ocr', False)  # Use PaddleOCR instead of DeepSeek-OCR

        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize required models for binning."""
        try:
            # Get optimal device distribution
            if self.enable_multi_gpu:
                device_map = self.gpu_manager.get_model_distribution()
                logger.info(f"Multi-GPU enabled with {self.gpu_manager.num_gpus} GPU(s)")
                logger.info(f"Model distribution: {device_map}")
            else:
                # Use single device
                single_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                device_map = {
                    'ocr': single_device,
                    'yolo': single_device,
                    'clip': single_device,
                    'blip': single_device
                }
                logger.info(f"Single device mode: {single_device}")

            # OCR for text detection - choose backend
            ocr_device = device_map['ocr']

            if self.use_paddle_ocr or not DEEPSEEK_AVAILABLE:
                # Use PaddleOCR (lightweight, ~200MB)
                if not PADDLE_AVAILABLE:
                    raise ImportError(
                        "PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle-gpu"
                    )

                logger.info(f"Loading PaddleOCR on {ocr_device}...")
                # Convert PyTorch device format to PaddleOCR format
                if ocr_device == "cpu":
                    paddle_device = "cpu"
                else:
                    # Convert cuda:0 to gpu:0
                    gpu_id = int(ocr_device.split(":")[-1]) if ":" in ocr_device else 0
                    paddle_device = f"gpu:{gpu_id}"

                self.ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang='en',
                    device=paddle_device,
                    enable_mkldnn=True if paddle_device == "cpu" else False,
                )
                self.ocr_backend = 'paddle'
                logger.info(f"PaddleOCR loaded successfully on {ocr_device}")

            else:
                # Try DeepSeek-OCR (heavy, ~10GB) with automatic fallback to PaddleOCR
                try:
                    logger.info(f"Loading DeepSeek-OCR on {ocr_device}...")
                    ocr_config = OCRConfig(
                        device=ocr_device,
                        crop_mode=True,
                        model_size=self.config.get('deepseek_model_size', 'tiny')
                    )
                    self.ocr = DeepSeekOCR(config=ocr_config)
                    self.ocr_backend = 'deepseek'
                    logger.info(f"✓ DeepSeek-OCR loaded successfully on {ocr_device}")

                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    # DeepSeek-OCR failed (likely OOM), fall back to PaddleOCR
                    logger.warning(f"✗ DeepSeek-OCR failed to load: {str(e)[:100]}...")
                    logger.warning("→ Falling back to PaddleOCR...")

                    if not PADDLE_AVAILABLE:
                        raise ImportError(
                            "DeepSeek-OCR failed and PaddleOCR not available. "
                            "Install PaddleOCR with: pip install paddleocr paddlepaddle-gpu"
                        )

                    # Clear any allocated memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Load PaddleOCR instead
                    # Convert PyTorch device format to PaddleOCR format
                    if ocr_device == "cpu":
                        paddle_device = "cpu"
                    else:
                        # Convert cuda:0 to gpu:0
                        gpu_id = int(ocr_device.split(":")[-1]) if ":" in ocr_device else 0
                        paddle_device = f"gpu:{gpu_id}"

                    self.ocr = PaddleOCR(
                        use_textline_orientation=True,
                        lang='en',
                        device=paddle_device,
                        enable_mkldnn=True if paddle_device == "cpu" else False,
                    )
                    self.ocr_backend = 'paddle'
                    logger.info(f"✓ PaddleOCR loaded successfully on {ocr_device} (fallback mode)")

            # Clear cache after loading OCR (frees up memory on other GPUs)
            if self.gpu_manager.num_gpus > 1:
                self.gpu_manager.clear_cache()

            # Object detection - YOLO or SAM
            detector_device = device_map['yolo']  # Use same device mapping for both detectors

            if self.object_detector == 'sam':
                # SAM-based object detection
                if not SAM_AVAILABLE:
                    logger.warning("SAM not available, falling back to YOLO")
                    self.object_detector = 'yolo'
                else:
                    logger.info(f"Loading SAM ({self.sam_model_type}) on {detector_device}...")
                    try:
                        # Load SAM model
                        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
                        if detector_device != "cpu":
                            sam.to(device=detector_device)

                        # Create automatic mask generator
                        self.sam_generator = SamAutomaticMaskGenerator(
                            model=sam,
                            points_per_side=32,
                            pred_iou_thresh=0.86,
                            stability_score_thresh=0.92,
                            crop_n_layers=1,
                            crop_n_points_downscale_factor=2,
                            min_mask_region_area=100,
                        )
                        self.detector_device = detector_device
                        logger.info(f"SAM loaded successfully on {detector_device}")
                    except Exception as e:
                        logger.error(f"Failed to load SAM: {e}")
                        logger.warning("Falling back to YOLO")
                        self.object_detector = 'yolo'

            if self.object_detector == 'yolo':
                # YOLO-based object detection
                logger.info(f"Loading YOLO ({self.yolo_model}) on {detector_device}...")

                # Model mapping
                model_mapping = {
                    'yolov8n': 'yolov8n.pt',
                    'yolov8s': 'yolov8s.pt',
                    'yolov9s': 'yolov9s.pt',
                    'yolov10s': 'yolov10s.pt',
                    'yolov11s': 'yolo11s.pt',
                }

                # Load the selected YOLO model
                model_file = model_mapping.get(self.yolo_model, 'yolov8n.pt')
                self.yolo = YOLO(model_file)
                if detector_device != "cpu":
                    self.yolo.to(detector_device)
                logger.info(f"YOLO {self.yolo_model} loaded on {detector_device}")

                # Store device for YOLO inference
                self.detector_device = detector_device

            # CLIP for caption similarity
            clip_device = device_map['clip']
            logger.info(f"Loading CLIP on {clip_device}...")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='openai'
            )
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            # Move CLIP to appropriate device
            if clip_device != "cpu":
                self.clip_model = self.clip_model.to(clip_device)
            self.clip_device = clip_device
            logger.info(f"CLIP loaded on {clip_device}")

            # BLIP for captioning - support both BLIP and BLIP-2
            blip_device = device_map['blip']
            logger.info(f"Loading BLIP on {blip_device}...")
            if self.use_blip2:
                # BLIP-2 for higher quality captions
                self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16 if "cuda" in blip_device else torch.float32
                )
                self.blip_model.to(blip_device)
                logger.info(f"BLIP-2 loaded on {blip_device}")
            else:
                # BLIP-base for faster processing
                self.blip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self.blip_model.to(blip_device)
                logger.info(f"BLIP-base loaded on {blip_device}")

            self.blip_device = blip_device

            logger.info("All models initialized successfully")

            # Print memory summary
            self.gpu_manager.print_memory_summary()

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def detect_text(self, image_path: str) -> Tuple[int, float]:
        """
        Detect text in image using configured OCR backend.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (number of text boxes, text area ratio)
        """
        try:
            # Get image dimensions
            img = Image.open(image_path)
            img_w, img_h = img.size
            img_area = img_w * img_h

            bboxes = []

            if self.ocr_backend == 'paddle':
                # PaddleOCR processing
                result = self.ocr.ocr(str(image_path))

                if result and result[0]:
                    for line in result[0]:
                        # PaddleOCR returns: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                        points = line[0]
                        # Convert to [x1, y1, x2, y2] format (top-left, bottom-right)
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                        bboxes.append(bbox)

            else:  # deepseek backend
                # Use grounding mode to get bounding boxes
                prompt = "<image>\n<|grounding|>Detect all text regions in this image."
                result = self.ocr.process(str(image_path), prompt=prompt)

                # Parse bounding boxes from result
                # DeepSeekOCR returns bounding boxes in the format [[x1,y1,x2,y2], ...]
                if hasattr(result, 'bounding_boxes') and result.bounding_boxes:
                    bboxes = result.bounding_boxes
                elif hasattr(result, 'text') and result.text:
                    # Parse bounding boxes from text output if available
                    # Pattern to match coordinates like [x1,y1,x2,y2]
                    pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
                    matches = re.findall(pattern, result.text)
                    if matches:
                        # Convert normalized coordinates (0-999) to actual pixels
                        bboxes = [
                            [int(x1) * img_w / 1000, int(y1) * img_h / 1000,
                             int(x2) * img_w / 1000, int(y2) * img_h / 1000]
                            for x1, y1, x2, y2 in matches
                        ]

            if not bboxes:
                return 0, 0.0

            # Count text boxes
            num_boxes = len(bboxes)

            # Calculate text area ratio
            text_area = 0
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                box_w = abs(x2 - x1)
                box_h = abs(y2 - y1)
                text_area += box_w * box_h

            text_area_ratio = text_area / img_area

            return num_boxes, text_area_ratio

        except Exception as e:
            logger.warning(f"Error detecting text in {image_path}: {e}")
            return 0, 0.0
    
    def detect_objects_sam(self, image_path: str) -> Tuple[int, int, float]:
        """
        Detect objects in image using SAM (Segment Anything Model).

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (total objects, estimated unique classes, spatial dispersion)
            Note: SAM doesn't classify objects, so unique_classes is estimated based on mask diversity
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]

            # Generate masks
            masks = self.sam_generator.generate(image)

            if not masks or len(masks) == 0:
                return 0, 0, 0.0

            num_objects = len(masks)

            # Estimate unique classes based on mask area diversity
            # Group masks by similar areas to estimate different object types
            areas = [mask['area'] for mask in masks]
            if areas:
                # Use clustering of areas as a proxy for unique object types
                area_bins = np.histogram(areas, bins=min(10, len(areas)))[0]
                unique_classes = np.count_nonzero(area_bins)
            else:
                unique_classes = 0

            # Calculate spatial dispersion using mask centroids
            if masks:
                centers = []
                for mask in masks:
                    # Get bounding box from mask
                    bbox = mask['bbox']  # Format: [x, y, w, h]
                    center_x = bbox[0] + bbox[2] / 2
                    center_y = bbox[1] + bbox[3] / 2
                    centers.append((center_x, center_y))

                if centers:
                    centers = np.array(centers)
                    # Calculate bounding box of all objects
                    min_x, min_y = np.min(centers, axis=0)
                    max_x, max_y = np.max(centers, axis=0)
                    dispersion = (max_x - min_x) * (max_y - min_y)
                    dispersion_ratio = dispersion / (img_w * img_h)
                else:
                    dispersion_ratio = 0.0
            else:
                dispersion_ratio = 0.0

            return num_objects, unique_classes, dispersion_ratio

        except Exception as e:
            logger.warning(f"Error detecting objects with SAM in {image_path}: {e}")
            return 0, 0, 0.0

    def detect_objects_yolo(self, image_path: str) -> Tuple[int, int, float]:
        """
        Detect objects in image using YOLO.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (total objects, unique classes, spatial dispersion)
        """
        try:
            # Run object detection
            results = self.yolo(str(image_path))

            if not results or len(results) == 0:
                return 0, 0, 0.0

            det = results[0]

            if det.boxes is None:
                return 0, 0, 0.0

            # Extract object information
            classes = det.boxes.cls.cpu().numpy().astype(int)
            boxes = det.boxes.xyxy.cpu().numpy()

            num_objects = len(classes)
            unique_classes = len(set(classes))

            # Calculate spatial dispersion
            if len(boxes) > 0:
                # Get centers of all boxes
                centers = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

                if centers:
                    centers = np.array(centers)
                    # Calculate bounding box of all objects
                    min_x, min_y = np.min(centers, axis=0)
                    max_x, max_y = np.max(centers, axis=0)
                    dispersion = (max_x - min_x) * (max_y - min_y)

                    # Normalize by image size
                    img = Image.open(image_path)
                    img_w, img_h = img.size
                    dispersion_ratio = dispersion / (img_w * img_h)
                else:
                    dispersion_ratio = 0.0
            else:
                dispersion_ratio = 0.0

            return num_objects, unique_classes, dispersion_ratio

        except Exception as e:
            logger.warning(f"Error detecting objects with YOLO in {image_path}: {e}")
            return 0, 0, 0.0

    def detect_objects(self, image_path: str) -> Tuple[int, int, float]:
        """
        Detect objects in image using the configured detector (YOLO or SAM).

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (total objects, unique classes, spatial dispersion)
        """
        if self.object_detector == 'sam':
            return self.detect_objects_sam(image_path)
        else:
            return self.detect_objects_yolo(image_path)
    
    def generate_caption(self, image_path: str) -> str:
        """
        Generate caption for image using BLIP or BLIP-2.

        Args:
            image_path: Path to the image file

        Returns:
            Generated caption text
        """
        try:
            image = Image.open(image_path).convert('RGB')

            if self.use_blip2:
                # BLIP-2 processing
                inputs = self.blip_processor(images=image, return_tensors="pt").to(
                    self.blip_device,
                    torch.float16 if "cuda" in self.blip_device else torch.float32
                )
                generated_ids = self.blip_model.generate(**inputs)
                caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            else:
                # BLIP-base processing
                inputs = self.blip_processor(image, return_tensors="pt").to(self.blip_device)
                out = self.blip_model.generate(**inputs, max_new_tokens=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)

            return caption
        except Exception as e:
            logger.warning(f"Error generating caption for {image_path}: {e}")
            return ""
    
    def calculate_clip_similarity(self, image_path: str, caption: str) -> float:
        """
        Calculate CLIP similarity between image and caption.

        Args:
            image_path: Path to the image file
            caption: Text caption to compare

        Returns:
            Cosine similarity score
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.clip_device)

            # Tokenize caption
            text_tokens = self.clip_tokenizer([caption]).to(self.clip_device)

            # Get embeddings
            with torch.no_grad():
                img_emb = self.clip_model.encode_image(img_tensor)
                text_emb = self.clip_model.encode_text(text_tokens)

                # Normalize embeddings
                img_emb /= img_emb.norm(dim=-1, keepdim=True)
                text_emb /= text_emb.norm(dim=-1, keepdim=True)

                # Calculate cosine similarity
                similarity = (img_emb @ text_emb.T).item()

            return similarity

        except Exception as e:
            logger.warning(f"Error calculating CLIP similarity for {image_path}: {e}")
            return 0.0
    
    def categorize_image(self, image_path: str) -> str:
        """
        Categorize a single image into bin A, B, or C.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Bin category ('A', 'B', or 'C')
        """
        # Check for text content (Bin A)
        num_text_boxes, text_area_ratio = self.detect_text(image_path)
        
        if (num_text_boxes > self.text_boxes_threshold or 
            text_area_ratio > self.text_area_threshold):
            return 'A'
        
        # Check for object content (Bin B)
        num_objects, unique_classes, dispersion = self.detect_objects(image_path)
        
        if (unique_classes > self.unique_objects_threshold or 
            num_objects > self.object_count_threshold):
            return 'B'
        
        # Check caption quality for Bin C
        caption = self.generate_caption(image_path)
        if caption:
            similarity = self.calculate_clip_similarity(image_path, caption)
            
            if similarity < self.clip_threshold:
                # Poor caption match, might not be suitable
                logger.debug(f"Image {image_path} has low CLIP similarity: {similarity}")
        
        # Default to Bin C (commonsense/attribute)
        return 'C'
    
    def bin_images(self, images: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize multiple images into bins.

        Args:
            images: List of image dictionaries with 'path' key

        Returns:
            Dictionary with bin categories as keys and image lists as values
        """
        # Log starting amount
        total_start = len(images)
        logger.info(f"Binning started with {total_start} images")

        bins = {'A': [], 'B': [], 'C': []}

        for img_data in images:
            try:
                bin_category = self.categorize_image(img_data['path'])
                bins[bin_category].append(img_data)

            except Exception as e:
                logger.error(f"Error binning image {img_data['path']}: {e}")
                # Default to Bin C on error
                bins['C'].append(img_data)

        # Log ending amount for each bin
        logger.info(f"Binning complete:")
        logger.info(f"  Started with: {total_start} images")
        logger.info(f"  Bin A (Text/Arithmetic): {len(bins['A'])} images")
        logger.info(f"  Bin B (Object/Spatial): {len(bins['B'])} images")
        logger.info(f"  Bin C (Commonsense/Attribute): {len(bins['C'])} images")
        logger.info(f"  Total binned: {len(bins['A']) + len(bins['B']) + len(bins['C'])} images")

        return bins
    
    def balance_bins(
        self,
        bins: Dict[str, List[Dict]],
        target_ratio: Tuple[float, float, float]
    ) -> Dict[str, List[Dict]]:
        """
        Balance bins according to target ratio.
        
        Args:
            bins: Dictionary of binned images
            target_ratio: Target ratio for (A, B, C) bins
            
        Returns:
            Balanced bins dictionary
        """
        total_images = sum(len(bin_images) for bin_images in bins.values())
        
        # Calculate target counts
        target_a = int(total_images * target_ratio[0])
        target_b = int(total_images * target_ratio[1])
        target_c = total_images - target_a - target_b
        
        # Create balanced bins
        balanced = {
            'A': bins['A'][:target_a] if len(bins['A']) >= target_a else bins['A'],
            'B': bins['B'][:target_b] if len(bins['B']) >= target_b else bins['B'],
            'C': bins['C'][:target_c] if len(bins['C']) >= target_c else bins['C']
        }
        
        # If any bin is under target, redistribute
        actual_total = sum(len(bin_images) for bin_images in balanced.values())
        
        if actual_total < total_images:
            # Add more from the bins that have extras
            shortage = total_images - actual_total
            
            for bin_key in ['A', 'B', 'C']:
                available = len(bins[bin_key]) - len(balanced[bin_key])
                if available > 0:
                    to_add = min(available, shortage)
                    start_idx = len(balanced[bin_key])
                    balanced[bin_key].extend(bins[bin_key][start_idx:start_idx + to_add])
                    shortage -= to_add
                    
                    if shortage <= 0:
                        break
        
        logger.info(f"Balanced bins - A: {len(balanced['A'])}, "
                   f"B: {len(balanced['B'])}, C: {len(balanced['C'])}")

        return balanced

    def filter_by_complexity(self, image_path: str) -> bool:
        """
        Filter images based on visual complexity criteria.

        Merged from yolov11-filter.py - an image passes if it has:
        - More than 3 unique object classes, OR
        - More than 5 total object instances, OR
        - Spatial dispersion > 0.3

        Args:
            image_path: Path to the image file

        Returns:
            True if image passes complexity filter, False otherwise
        """
        num_objects, unique_classes, dispersion = self.detect_objects(image_path)

        passes = (
            unique_classes > self.unique_objects_threshold or
            num_objects > self.object_count_threshold or
            dispersion > self.spatial_dispersion_threshold
        )

        logger.debug(
            f"Complexity check for {image_path}: "
            f"objects={num_objects}, classes={unique_classes}, "
            f"dispersion={dispersion:.3f}, passes={passes}"
        )

        return passes

    def benchmark_yolo_models(self, images: List[str]) -> pd.DataFrame:
        """
        Benchmark multiple YOLO models on a set of images.

        Merged from yolov11-filter.py - tests all configured YOLO models
        and returns performance metrics including inference time and
        filtering pass rates.

        Args:
            images: List of image file paths

        Returns:
            DataFrame with benchmark results for all models
        """
        if not self.enable_multi_yolo:
            logger.warning("Multi-YOLO not enabled. Enable with enable_multi_yolo=True in config.")
            return pd.DataFrame()

        all_results = []

        for model_name, yolo_model in self.yolo_models.items():
            logger.info(f"Benchmarking {model_name}...")

            results_list = []
            times = []

            for img_path in images:
                start = time.time()
                results = yolo_model(img_path, verbose=False)[0]
                elapsed = time.time() - start
                times.append(elapsed)

                boxes = results.boxes
                classes = boxes.cls.cpu().numpy() if len(boxes) > 0 else []
                centers = boxes.xywh[:, :2].cpu().numpy() if len(boxes) > 0 else np.array([])

                unique_classes = len(np.unique(classes))
                total_instances = len(boxes)

                # Spatial dispersion
                if len(centers) > 1:
                    img_diag = np.sqrt(results.orig_shape[0]**2 + results.orig_shape[1]**2)
                    spatial_dispersion = np.std(centers, axis=0).mean() / img_diag
                else:
                    spatial_dispersion = 0

                # Check filtering criteria
                passes_filter = (
                    unique_classes > self.unique_objects_threshold or
                    total_instances > self.object_count_threshold or
                    spatial_dispersion > self.spatial_dispersion_threshold
                )

                results_list.append({
                    'image': Path(img_path).name,
                    'model_type': 'YOLO',
                    'model_name': model_name,
                    'unique_classes': unique_classes,
                    'total_instances': total_instances,
                    'spatial_dispersion': spatial_dispersion,
                    'passes_filter': passes_filter,
                    'inference_time': elapsed
                })

            avg_time = np.mean(times)
            total_time = sum(times)
            pass_count = sum(r['passes_filter'] for r in results_list)

            logger.info(
                f"{model_name} - Avg time: {avg_time:.3f}s, "
                f"Total: {total_time:.3f}s, "
                f"Pass rate: {pass_count}/{len(results_list)}"
            )

            # Add average time to all results
            for r in results_list:
                r['avg_inference_time'] = avg_time

            all_results.extend(results_list)

        df_results = pd.DataFrame(all_results)
        logger.info("YOLO benchmarking complete")

        return df_results

    def generate_captions_batch(self, images: List[str]) -> List[str]:
        """
        Generate captions for multiple images.

        Merged from blip2.py - batch caption generation with error handling.

        Args:
            images: List of image file paths

        Returns:
            List of generated captions
        """
        captions = []

        for i, img_path in enumerate(images, 1):
            logger.info(f"Processing image #{i}: {Path(img_path).name}")

            try:
                caption = self.generate_caption(img_path)
                captions.append(caption)
                logger.info(f"Caption generated: {caption}")

            except Exception as e:
                logger.error(f"ERROR occurred processing {img_path}: {e}")
                captions.append("")

        logger.info(f"All {len(images)} images have been processed")
        return captions
