"""
Image binning module for categorizing images into Text/Object/Commonsense bins.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import open_clip
from deep_ocr import DeepSeekOCR, OCRConfig
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
import pandas as pd
import time
import re

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
        self.enable_multi_yolo = config.get('enable_multi_yolo', False)
        self.yolo_models_list = config.get('yolo_models', ['yolov8n'])

        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize required models for binning."""
        try:
            # OCR for text detection - using DeepSeekOCR
            ocr_config = OCRConfig(
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                crop_mode=True,
                model_size=self.config.get('deepseek_model_size', 'tiny')  # Use 'tiny' by default for memory efficiency
            )
            self.ocr = DeepSeekOCR(config=ocr_config)

            # Object detection - Multi-YOLO support
            if self.enable_multi_yolo:
                self.yolo_models = {}
                model_mapping = {
                    'yolov8n': 'yolov8n.pt',
                    'yolov8s': 'yolov8s.pt',
                    'yolov9s': 'yolov9s.pt',
                    'yolov10s': 'yolov10s.pt',
                    'yolov11s': 'yolo11s.pt',
                }
                for model_name in self.yolo_models_list:
                    if model_name in model_mapping:
                        self.yolo_models[model_name] = YOLO(model_mapping[model_name])
                        logger.info(f"Loaded {model_name}")

                # Set default YOLO model
                self.yolo = self.yolo_models[self.yolo_models_list[0]]
            else:
                # Single YOLO model (default: v8n for speed)
                self.yolo = YOLO('yolov8n.pt')

            # CLIP for caption similarity
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='openai'
            )
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

            # BLIP for captioning - support both BLIP and BLIP-2
            if self.use_blip2:
                # BLIP-2 for higher quality captions
                self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
                self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.blip_model.to(self.device)
                logger.info("Loaded BLIP-2 model")
            else:
                # BLIP-base for faster processing
                self.device = torch.device("cpu")
                self.blip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                logger.info("Loaded BLIP-base model")

            logger.info("All models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def detect_text(self, image_path: str) -> Tuple[int, float]:
        """
        Detect text in image using DeepSeekOCR.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (number of text boxes, text area ratio)
        """
        try:
            # Use grounding mode to get bounding boxes
            prompt = "<image>\n<|grounding|>Detect all text regions in this image."
            result = self.ocr.process(str(image_path), prompt=prompt)

            # Get image dimensions
            img = Image.open(image_path)
            img_w, img_h = img.size
            img_area = img_w * img_h

            # Parse bounding boxes from result
            # DeepSeekOCR returns bounding boxes in the format [[x1,y1,x2,y2], ...]
            bboxes = []
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
    
    def detect_objects(self, image_path: str) -> Tuple[int, int, float]:
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
            logger.warning(f"Error detecting objects in {image_path}: {e}")
            return 0, 0, 0.0
    
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
                    self.device,
                    torch.float16 if torch.cuda.is_available() else torch.float32
                )
                generated_ids = self.blip_model.generate(**inputs)
                caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            else:
                # BLIP-base processing
                inputs = self.blip_processor(image, return_tensors="pt")
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
            img_tensor = self.clip_preprocess(image).unsqueeze(0)
            
            # Tokenize caption
            text_tokens = self.clip_tokenizer([caption])
            
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
        bins = {'A': [], 'B': [], 'C': []}
        
        for img_data in images:
            try:
                bin_category = self.categorize_image(img_data['path'])
                bins[bin_category].append(img_data)
                
            except Exception as e:
                logger.error(f"Error binning image {img_data['path']}: {e}")
                # Default to Bin C on error
                bins['C'].append(img_data)
        
        logger.info(f"Binning complete - A: {len(bins['A'])}, "
                   f"B: {len(bins['B'])}, C: {len(bins['C'])}")
        
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
