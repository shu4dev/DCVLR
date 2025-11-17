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
from paddleocr import PaddleOCR
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

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
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize required models for binning."""
        try:
            # OCR for text detection
            self.ocr = PaddleOCR(
                lang='en',
                use_angle_cls=True,
                show_log=False
            )
            
            # Object detection
            self.yolo = YOLO('yolov8n.pt')
            
            # CLIP for caption similarity
            self.clip_model, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='openai'
            )
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            # BLIP for captioning
            self.blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def detect_text(self, image_path: str) -> Tuple[int, float]:
        """
        Detect text in image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (number of text boxes, text area ratio)
        """
        try:
            # Run OCR detection only (no recognition for speed)
            result = self.ocr.ocr(str(image_path), det=True, rec=False, cls=False)
            
            if not result or not result[0]:
                return 0, 0.0
            
            # Count text boxes
            num_boxes = len(result[0])
            
            # Calculate text area ratio
            img = Image.open(image_path)
            img_w, img_h = img.size
            img_area = img_w * img_h
            
            text_area = 0
            for box_info in result[0]:
                box = box_info[0] if isinstance(box_info, (list, tuple)) else box_info
                # Box format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                box_w = max(x_coords) - min(x_coords)
                box_h = max(y_coords) - min(y_coords)
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
        Generate caption for image using BLIP.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Generated caption text
        """
        try:
            image = Image.open(image_path).convert('RGB')
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
