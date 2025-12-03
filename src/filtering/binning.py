"""
Image binning module for categorizing images into Text/Object/Commonsense bins.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
import tempfile
import shutil

import numpy as np
from PIL import Image
import torch
import open_clip
import pandas as pd
import time
import re
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.gpu_utils import GPUManager

# Setup logger early
logger = logging.getLogger(__name__)

# Check transformers version for compatibility
try:
    import transformers
    from packaging import version
    TRANSFORMERS_VERSION = version.parse(transformers.__version__)
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_VERSION = None
    TRANSFORMERS_AVAILABLE = False
    logger.warning("packaging library not available, version checking disabled")

# DeepSeek-OCR requires transformers==4.46.3
try:
    from transformers import AutoModel, AutoTokenizer
    if TRANSFORMERS_VERSION and TRANSFORMERS_VERSION >= version.parse("4.46.3"):
        DEEPSEEK_AVAILABLE = True
    else:
        DEEPSEEK_AVAILABLE = False
        if TRANSFORMERS_AVAILABLE and TRANSFORMERS_VERSION:
            logger.warning(f"DeepSeek-OCR requires transformers==4.46.3, found {transformers.__version__}")
except ImportError:
    DEEPSEEK_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

from ultralytics import YOLO

# BLIP models require transformers>=4.47
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
    BLIP_AVAILABLE = True
    if TRANSFORMERS_VERSION and TRANSFORMERS_VERSION < version.parse("4.47.0"):
        logger.warning(f"BLIP models work best with transformers>=4.47.0, found {transformers.__version__}")
except ImportError:
    BLIP_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


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

        # Pipeline mode selection
        self.pipeline_mode = config.get('pipeline_mode', 'hybrid').lower()  # 'hybrid' or 'deepseek_unified'

        # Model configuration
        self.use_blip2 = config.get('use_blip2', False)

        # Captioning backend selection (only for hybrid mode)
        self.captioner_backend = config.get('captioner_backend', 'blip').lower()  # 'blip', 'blip2', or 'moondream'
        self.moondream_api_key = config.get('moondream_api_key', None)
        self.moondream_caption_length = config.get('moondream_caption_length', 'normal')

        # Object detector selection (only for hybrid mode)
        self.object_detector = config.get('object_detector', 'yolo').lower()  # 'yolo' or 'sam'

        # YOLO-specific configuration
        self.yolo_model = config.get('yolo_model', 'yolov8n')

        # SAM-specific configuration
        self.sam_model_type = config.get('sam_model_type', 'vit_b')
        self.sam_checkpoint = config.get('sam_checkpoint', 'models/sam_vit_b_01ec64.pth')

        # GPU configuration
        self.enable_multi_gpu = config.get('enable_multi_gpu', True)  # Auto-detect and use multiple GPUs
        self.gpu_manager = GPUManager()

        # Initialize models based on pipeline mode
        self._init_models()
    
    def _init_models(self):
        """Initialize required models for binning based on pipeline mode."""
        try:
            logger.info(f"Initializing models in '{self.pipeline_mode}' pipeline mode...")

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

            # Initialize based on pipeline mode
            if self.pipeline_mode == 'deepseek_unified':
                self._init_deepseek_unified_pipeline(device_map)
            else:  # hybrid mode (default)
                self._init_hybrid_pipeline(device_map)

            logger.info("All models initialized successfully")

            # Print memory summary
            self.gpu_manager.print_memory_summary()

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def _init_hybrid_pipeline(self, device_map: Dict[str, str]):
        """Initialize models for hybrid pipeline mode (PaddleOCR + separate models)."""
        logger.info("Setting up HYBRID pipeline (PaddleOCR + YOLO/SAM + BLIP/BLIP2/Moondream)")

        # Check transformers version for BLIP models
        if self.captioner_backend in ['blip', 'blip2']:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers library not available. Install with: pip install transformers>=4.47.0"
                )
            if TRANSFORMERS_VERSION and TRANSFORMERS_VERSION < version.parse("4.47.0"):
                logger.warning(
                    f"BLIP models work best with transformers>=4.47.0, found {transformers.__version__}. "
                    f"Consider upgrading: pip install --upgrade transformers"
                )

        # OCR for text detection - always use PaddleOCR in hybrid mode
        ocr_device = device_map['ocr']

        # Use PaddleOCR (lightweight, ~200MB) - hybrid mode default
        if not PADDLE_AVAILABLE:
            raise ImportError(
                "PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle-gpu"
            )

        logger.info(f"Loading PaddleOCR on {ocr_device}...")

        # PaddleOCR new API - device is handled automatically based on GPU availability
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        self.ocr_backend = 'paddle'
        self.ocr_device = ocr_device
        logger.info(f"PaddleOCR loaded successfully on {ocr_device}")

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
                    # Load SAM model directly on target device
                    sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
                    # SAM loads on CPU by default, move immediately if GPU is target
                    if detector_device != "cpu":
                        sam = sam.to(device=detector_device)

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
            # YOLO loads on CPU by default
            if detector_device != "cpu":
                # Use device parameter during YOLO initialization if available, or move after
                self.yolo = YOLO(model_file)
                self.yolo.to(detector_device)
            else:
                self.yolo = YOLO(model_file)
            logger.info(f"YOLO {self.yolo_model} loaded on {detector_device}")

            # Store device for YOLO inference
            self.detector_device = detector_device

        # CLIP for caption similarity
        clip_device = device_map['clip']
        logger.info(f"Loading CLIP on {clip_device}...")
        # Load CLIP with device parameter if supported, otherwise move after
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai',
            device=clip_device if clip_device != "cpu" else None
        )
        if clip_device != "cpu":
            self.clip_model = self.clip_model.to(clip_device)

        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip_device = clip_device

        logger.info(f"CLIP loaded on {clip_device}")

        # Captioning backend - support BLIP, BLIP-2, or Moondream
        if self.captioner_backend == 'moondream':
            # Moondream API-based captioning
            if not self.moondream_api_key:
                logger.warning("Moondream API key not provided, falling back to BLIP")
                self.captioner_backend = 'blip'
            else:
                logger.info("Loading Moondream API captioner...")
                try:
                    from src.synthesis.moondream_captioner import MoondreamCaptioner
                    self.moondream_captioner = MoondreamCaptioner(
                        api_key=self.moondream_api_key,
                        length=self.moondream_caption_length
                    )
                    logger.info(f"Moondream API captioner loaded (length={self.moondream_caption_length})")
                    self.blip_model = None
                    self.blip_processor = None
                    self.blip_device = None
                except Exception as e:
                    logger.error(f"Failed to load Moondream captioner: {e}")
                    logger.warning("Falling back to BLIP")
                    self.captioner_backend = 'blip'

        if self.captioner_backend in ['blip', 'blip2']:
            # BLIP/BLIP-2 for local captioning
            blip_device = device_map['blip']
            logger.info(f"Loading BLIP on {blip_device}...")

            if self.captioner_backend == 'blip2' or self.use_blip2:
                # BLIP-2 for higher quality captions
                self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

                # Set pad_token if not already set
                if self.blip_processor.tokenizer.pad_token is None:
                    self.blip_processor.tokenizer.pad_token = self.blip_processor.tokenizer.eos_token
                    self.blip_processor.tokenizer.pad_token_id = self.blip_processor.tokenizer.eos_token_id

                # Determine dtype and device for loading
                model_dtype = torch.float16 if "cuda" in blip_device else torch.float32

                # Load model without device_map for better compatibility
                self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=model_dtype
                )
                # Move model to device explicitly
                if "cuda" in blip_device:
                    self.blip_model = self.blip_model.to(blip_device)
                self.blip_model.eval()
                logger.info(f"BLIP-2 loaded on {blip_device}")
                self.captioner_backend = 'blip2'
            else:
                # BLIP-base for faster processing
                self.blip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )

                # Set pad_token if not already set
                if self.blip_processor.tokenizer.pad_token is None:
                    self.blip_processor.tokenizer.pad_token = self.blip_processor.tokenizer.eos_token
                    self.blip_processor.tokenizer.pad_token_id = self.blip_processor.tokenizer.eos_token_id

                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self.blip_model.to(blip_device)
                logger.info(f"BLIP-base loaded on {blip_device}")
                self.captioner_backend = 'blip'

            self.blip_device = blip_device

    def _init_deepseek_unified_pipeline(self, device_map: Dict[str, str]):
        """Initialize models for DeepSeek unified pipeline mode (DeepSeek-OCR for all tasks)."""
        logger.info("Setting up DEEPSEEK UNIFIED pipeline (DeepSeek-OCR for OCR + Object Detection + Captioning)")

        # Single DeepSeek-OCR model for all tasks
        ocr_device = device_map['ocr']

        # Check transformers version
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library not available. Install with: pip install transformers==4.46.3"
            )

        if TRANSFORMERS_VERSION < version.parse("4.46.3"):
            raise ImportError(
                f"DeepSeek-OCR requires transformers==4.46.3, but found {transformers.__version__}. "
                f"Please install the correct version: pip install transformers==4.46.3"
            )

        if TRANSFORMERS_VERSION > version.parse("4.46.3"):
            logger.warning(
                f"DeepSeek-OCR works best with transformers==4.46.3, but found {transformers.__version__}. "
                f"You may encounter compatibility issues. Consider downgrading: pip install transformers==4.46.3"
            )

        if not DEEPSEEK_AVAILABLE:
            raise ImportError(
                "DeepSeek-OCR not available. Install with: pip install transformers==4.46.3"
            )

        logger.info(f"Loading DeepSeek-OCR on {ocr_device} for unified pipeline...")

        # Load model and tokenizer from HuggingFace
        model_name = 'deepseek-ai/DeepSeek-OCR'
        self.ocr_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Determine dtype based on device
        model_dtype = torch.bfloat16 if ocr_device != "cpu" else torch.float32

        self.ocr_model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation='flash_attention_2',
            torch_dtype=model_dtype,
            device_map=ocr_device if ocr_device != "cpu" else None,
            trust_remote_code=True,
            use_safetensors=True
        )

        # Ensure model is on correct device if device_map didn't work
        if ocr_device != "cpu" and self.ocr_model.device.type == 'cpu':
            self.ocr_model = self.ocr_model.to(ocr_device)

        self.ocr_model = self.ocr_model.eval()
        self.ocr_device = ocr_device

        # Set model size parameters based on config
        model_size = self.config.get('deepseek_model_size', 'tiny').lower()
        if model_size == 'tiny':
            self.ocr_base_size = 512
            self.ocr_image_size = 512
            self.ocr_crop_mode = False
        elif model_size == 'small':
            self.ocr_base_size = 640
            self.ocr_image_size = 640
            self.ocr_crop_mode = False
        elif model_size == 'base':
            self.ocr_base_size = 1024
            self.ocr_image_size = 1024
            self.ocr_crop_mode = False
        elif model_size == 'large':
            self.ocr_base_size = 1280
            self.ocr_image_size = 1280
            self.ocr_crop_mode = False
        else:  # gundam
            self.ocr_base_size = 1024
            self.ocr_image_size = 640
            self.ocr_crop_mode = True

        self.ocr_backend = 'deepseek'
        logger.info(f"✓ DeepSeek-OCR loaded successfully on {ocr_device} (size: {model_size})")

        # In unified mode, we use DeepSeek for all tasks
        self.object_detector = 'deepseek'
        self.captioner_backend = 'deepseek'
        self.detector_device = ocr_device

        # Set dummy values for unused models
        self.yolo = None
        self.sam_generator = None
        self.blip_model = None
        self.blip_processor = None
        self.blip_device = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.clip_device = None
        self.moondream_captioner = None

        logger.info("DeepSeek unified pipeline initialized successfully")
    
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
                # PaddleOCR processing - using new predict() API
                results = self.ocr.predict(input=str(image_path))

                if results:
                    for result in results:
                        # New API returns result objects with structured data
                        # Access the OCR results from the result object
                        if hasattr(result, 'json') and result.json:
                            data_json = result.json
                            # Handle both dict and list formats
                            if isinstance(data_json, dict):
                                # If dict, iterate through values looking for list of items
                                for key, value in data_json.items():
                                    if isinstance(value, (list, tuple)):
                                        for item in value:
                                            if isinstance(item, dict) and 'points' in item:
                                                # Extract bounding box points
                                                points = item['points']
                                                # Convert to [x1, y1, x2, y2] format (top-left, bottom-right)
                                                xs = [p[0] for p in points]
                                                ys = [p[1] for p in points]
                                                bbox = [min(xs), min(ys), max(xs), max(ys)]
                                                bboxes.append(bbox)
                            elif isinstance(data_json, (list, tuple)):
                                # If list, directly iterate through items
                                for item in data_json:
                                    if isinstance(item, dict) and 'points' in item:
                                        # Extract bounding box points
                                        points = item['points']
                                        # Convert to [x1, y1, x2, y2] format (top-left, bottom-right)
                                        xs = [p[0] for p in points]
                                        ys = [p[1] for p in points]
                                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                                        bboxes.append(bbox)

            else:  # deepseek backend
                # Use grounding mode to detect text regions
                prompt = "<image>\n<|grounding|>Convert the document to markdown."

                # Create a temporary directory for output files (DeepSeek-OCR requires this)
                temp_dir = tempfile.mkdtemp(prefix='deepseek_ocr_')

                # Call the infer method with proper parameters
                try:
                    result = self.ocr_model.infer(
                        self.ocr_tokenizer,
                        prompt=prompt,
                        image_file=str(image_path),
                        output_path=temp_dir,  # Use temp directory
                        base_size=self.ocr_base_size,
                        image_size=self.ocr_image_size,
                        crop_mode=self.ocr_crop_mode,
                        save_results=False,
                        test_compress=False
                    )
                except Exception as ocr_error:
                    # Catch errors from DeepSeek OCR
                    logger.warning(f"DeepSeek OCR failed for {image_path}: {ocr_error}")
                    return 0, 0.0
                finally:
                    # Clean up temporary directory
                    shutil.rmtree(temp_dir, ignore_errors=True)

                # Parse the result text to extract bounding boxes
                # DeepSeek OCR with grounding mode returns text with coordinates
                if result and isinstance(result, str):
                    # Pattern to match coordinates like [x1,y1,x2,y2]
                    pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
                    matches = re.findall(pattern, result)
                    if matches:
                        # Filter out invalid bboxes (e.g., [0,0,999,999] which covers entire image)
                        valid_matches = []
                        for x1, y1, x2, y2 in matches:
                            # Skip bboxes that cover the entire normalized space
                            if not (int(x1) == 0 and int(y1) == 0 and int(x2) == 999 and int(y2) == 999):
                                valid_matches.append((x1, y1, x2, y2))

                        if valid_matches:
                            # Convert normalized coordinates (0-999) to actual pixels
                            bboxes = [
                                [int(x1) * img_w / 1000, int(y1) * img_h / 1000,
                                 int(x2) * img_w / 1000, int(y2) * img_h / 1000]
                                for x1, y1, x2, y2 in valid_matches
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

            # Avoid division by zero
            text_area_ratio = text_area / img_area if img_area > 0 else 0.0

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

    def detect_objects_deepseek(self, image_path: str) -> Tuple[int, int, float]:
        """
        Detect objects in image using DeepSeek-OCR grounding mode.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (total objects, unique classes, spatial dispersion)
        """
        try:
            # Get image dimensions
            img = Image.open(image_path)
            img_w, img_h = img.size

            # Use grounding mode to detect objects
            prompt = "<image>\n<|grounding|>Detect all objects in this image."

            # Create a temporary directory for output files
            temp_dir = tempfile.mkdtemp(prefix='deepseek_obj_')

            try:
                result = self.ocr_model.infer(
                    self.ocr_tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=temp_dir,
                    base_size=self.ocr_base_size,
                    image_size=self.ocr_image_size,
                    crop_mode=self.ocr_crop_mode,
                    save_results=False,
                    test_compress=False
                )
            except Exception as ocr_error:
                logger.warning(f"DeepSeek object detection failed for {image_path}: {ocr_error}")
                return 0, 0, 0.0
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            # Parse result to extract bounding boxes
            bboxes = []
            if result and isinstance(result, str):
                # Pattern to match coordinates like [x1,y1,x2,y2]
                pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
                matches = re.findall(pattern, result)
                if matches:
                    # Filter out invalid bboxes
                    for x1, y1, x2, y2 in matches:
                        if not (int(x1) == 0 and int(y1) == 0 and int(x2) == 999 and int(y2) == 999):
                            # Convert normalized coordinates to pixels
                            bbox = [
                                int(x1) * img_w / 1000,
                                int(y1) * img_h / 1000,
                                int(x2) * img_w / 1000,
                                int(y2) * img_h / 1000
                            ]
                            bboxes.append(bbox)

            if not bboxes:
                return 0, 0, 0.0

            num_objects = len(bboxes)

            # Estimate unique classes based on bbox size diversity
            areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
            if areas:
                area_bins = np.histogram(areas, bins=min(10, len(areas)))[0]
                unique_classes = np.count_nonzero(area_bins)
            else:
                unique_classes = 0

            # Calculate spatial dispersion
            if bboxes:
                centers = []
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

                if centers:
                    centers = np.array(centers)
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
            logger.warning(f"Error detecting objects with DeepSeek in {image_path}: {e}")
            return 0, 0, 0.0

    def detect_objects(self, image_path: str) -> Tuple[int, int, float]:
        """
        Detect objects in image using the configured detector (YOLO, SAM, or DeepSeek).

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (total objects, unique classes, spatial dispersion)
        """
        if self.object_detector == 'sam':
            return self.detect_objects_sam(image_path)
        elif self.object_detector == 'deepseek':
            return self.detect_objects_deepseek(image_path)
        else:
            return self.detect_objects_yolo(image_path)

    def generate_caption_deepseek(self, image_path: str) -> str:
        """
        Generate caption for image using DeepSeek-OCR.

        Args:
            image_path: Path to the image file

        Returns:
            Generated caption text
        """
        try:
            # Use DeepSeek-OCR for caption generation
            prompt = "<image>\nDescribe this image in detail."

            # Create a temporary directory for output files
            temp_dir = tempfile.mkdtemp(prefix='deepseek_cap_')

            try:
                result = self.ocr_model.infer(
                    self.ocr_tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=temp_dir,
                    base_size=self.ocr_base_size,
                    image_size=self.ocr_image_size,
                    crop_mode=self.ocr_crop_mode,
                    save_results=False,
                    test_compress=False
                )
            except Exception as ocr_error:
                logger.warning(f"DeepSeek caption generation failed for {image_path}: {ocr_error}")
                return ""
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            # Extract caption from result
            if result and isinstance(result, str):
                # Remove any coordinate tags [x,y,x,y] from the result
                caption = re.sub(r'\[\d+,\d+,\d+,\d+\]', '', result).strip()
                return caption
            return ""

        except Exception as e:
            logger.warning(f"Error generating caption with DeepSeek for {image_path}: {e}")
            return ""

    def generate_caption(self, image_path: str) -> str:
        """
        Generate caption for image using BLIP, BLIP-2, Moondream, or DeepSeek.

        Args:
            image_path: Path to the image file

        Returns:
            Generated caption text
        """
        try:
            # Use DeepSeek-OCR if in unified mode
            if self.captioner_backend == 'deepseek':
                return self.generate_caption_deepseek(image_path)

            # Use Moondream API if configured
            if self.captioner_backend == 'moondream':
                caption = self.moondream_captioner.generate_caption(image_path)
                return caption if caption else ""

            # Use BLIP/BLIP-2 for local processing
            image = Image.open(image_path).convert('RGB')

            if self.captioner_backend == 'blip2':
                # BLIP-2 processing
                inputs = self.blip_processor(images=image, return_tensors="pt").to(
                    self.blip_device,
                    torch.float16 if "cuda" in self.blip_device else torch.float32
                )

                # Check if inputs are valid (not empty)
                if 'pixel_values' in inputs and inputs['pixel_values'].numel() > 0:
                    generated_ids = self.blip_model.generate(
                        **inputs,
                        pad_token_id=self.blip_processor.tokenizer.pad_token_id,
                        attention_mask=inputs.get('attention_mask', None)
                    )
                    caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                else:
                    logger.warning(f"Empty input tensors for {image_path}, skipping caption generation")
                    return ""
            else:
                # BLIP-base processing
                inputs = self.blip_processor(image, return_tensors="pt").to(self.blip_device)

                # Check if inputs are valid (not empty)
                if 'pixel_values' in inputs and inputs['pixel_values'].numel() > 0:
                    out = self.blip_model.generate(
                        **inputs,
                        max_new_tokens=50,
                        pad_token_id=self.blip_processor.tokenizer.pad_token_id,
                        attention_mask=inputs.get('attention_mask', None)
                    )
                    caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                else:
                    logger.warning(f"Empty input tensors for {image_path}, skipping caption generation")
                    return ""

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
            Cosine similarity score (or 1.0 if CLIP is not available in deepseek_unified mode)
        """
        # In deepseek_unified mode, CLIP is not available - skip similarity check
        if self.clip_model is None:
            logger.debug(f"CLIP not available (deepseek_unified mode), skipping similarity check")
            return 1.0  # Return high similarity to pass the check

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

    @staticmethod
    def load_images_from_train_folders(data_dir: str) -> List[Dict]:
        """
        Load all images from train folders within subdirectories of data_dir.

        Expected structure:
        data_dir/
            folder1/
                train/
                    image1.jpg
                    image2.png
                    ...
            folder2/
                train/
                    image3.jpg
                    ...
            ...

        Args:
            data_dir: Path to the data directory containing subdirectories with train folders

        Returns:
            List of dictionaries with image metadata:
            [{'path': '/path/to/image.jpg', 'dataset': 'folder1', 'filename': 'image1.jpg'}, ...]
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return []

        if not data_path.is_dir():
            logger.error(f"Data path is not a directory: {data_dir}")
            return []

        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

        images = []

        # Iterate through subdirectories in data_dir
        for subfolder in data_path.iterdir():
            if not subfolder.is_dir():
                continue

            # Look for train folder within this subfolder
            train_folder = subfolder / 'train'

            if not train_folder.exists() or not train_folder.is_dir():
                logger.warning(f"No train folder found in {subfolder.name}, skipping...")
                continue

            logger.info(f"Loading images from {subfolder.name}/train/")

            # Load all images from train folder
            img_count = 0
            for img_path in train_folder.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    images.append({
                        'path': str(img_path.absolute()),
                        'dataset': subfolder.name,
                        'filename': img_path.name
                    })
                    img_count += 1

            logger.info(f"  Loaded {img_count} images from {subfolder.name}/train/")

        logger.info(f"Total images loaded: {len(images)} from {len(set(img['dataset'] for img in images))} datasets")

        return images
