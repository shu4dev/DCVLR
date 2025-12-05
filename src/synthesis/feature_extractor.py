"""
Feature extraction module for the Team-1 Data Synthesis Pipeline.
Handles OCR, object detection, and image captioning.
"""

import logging
from typing import Dict, List, Any, Optional
from src.utils.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts various features from images including:
    - OCR text
    - Object detection
    - Image captioning
    - Spatial relationships
    """

    def __init__(self, device: str = "cuda", caption_only: bool = False):
        """
        Initialize feature extractors.

        Args:
            device: Device to run models on ('cuda' or 'cpu')
            caption_only: If True, only load captioning model (skip OCR and object detection)
        """
        self.device = device
        self.caption_only = caption_only
        self.ocr_model = None
        self.object_detector = None
        self.captioner = None

        logger.info(f"FeatureExtractor initialized (caption_only={caption_only})")
        self._load_models()

    def _load_models(self):
        """Load all feature extraction models."""
        registry = ModelRegistry.get_instance()
        
        # In caption-only mode, skip OCR and object detection
        if self.caption_only:
            logger.info("Caption-only mode: Skipping OCR and object detection models")
            self.ocr_model = None
            self.object_detector = None
        else:
            # Load OCR model
            try:
                def load_easyocr():
                    import easyocr
                    return easyocr.Reader(['en'], gpu=self.device == 'cuda')
                
                self.ocr_model = registry.get_model(f'easyocr_{self.device}', load_easyocr)
            except Exception as e:
                logger.warning(f"Could not load OCR model: {e}")
                self.ocr_model = None

            # Load object detector (YOLO)
            try:
                def load_yolo():
                    from ultralytics import YOLO
                    return YOLO('yolov8n.pt')
                
                self.object_detector = registry.get_model(f'yolo_yolov8n_{self.device}', load_yolo)
            except Exception as e:
                logger.warning(f"Could not load object detector: {e}")
                self.object_detector = None

        # Always load image captioner
        try:
            def load_captioner():
                from transformers import pipeline
                return pipeline(
                    "image-to-text",
                    model="Salesforce/blip-image-captioning-base",
                    device=0 if self.device == "cuda" else -1
                )
            
            self.captioner = registry.get_model(f'blip_captioner_{self.device}', load_captioner)
        except Exception as e:
            logger.warning(f"Could not load captioner: {e}")
            self.captioner = None

    def extract_ocr(self, image_path: str) -> str:
        """
        Extract text from image using OCR.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text string
        """
        if not self.ocr_model:
            return ""

        try:
            results = self.ocr_model.readtext(image_path)
            text = ' '.join([r[1] for r in results])
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def extract_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect objects in the image.

        Args:
            image_path: Path to the image file

        Returns:
            List of detected objects with bounding boxes
        """
        if not self.object_detector:
            return []

        try:
            results = self.object_detector(image_path, verbose=False)

            objects = []
            for result in results:
                for box in result.boxes:
                    obj = {
                        'class': result.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist()
                    }
                    objects.append(obj)

            return objects
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []

    def extract_caption(self, image_path: str) -> str:
        """
        Generate a caption for the image.

        Args:
            image_path: Path to the image file

        Returns:
            Generated caption string
        """
        if not self.captioner:
            return ""

        try:
            from PIL import Image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            result = self.captioner(img)
            if result and len(result) > 0:
                return result[0].get('generated_text', '')
            return ""
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return ""

    def extract_spatial_relations(
        self,
        objects: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Determine spatial relationships between objects.

        Args:
            objects: List of detected objects with bounding boxes

        Returns:
            List of spatial relationship descriptions
        """
        relations = []

        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                relation = self._get_spatial_relation(obj1, obj2)
                if relation:
                    relations.append(relation)

        return relations

    def _get_spatial_relation(
        self,
        obj1: Dict[str, Any],
        obj2: Dict[str, Any]
    ) -> Optional[str]:
        """Determine spatial relation between two objects."""
        bbox1 = obj1['bbox']
        bbox2 = obj2['bbox']

        # Calculate centers
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2

        # Determine relation
        if center1_x < center2_x - 50:
            h_rel = "left of"
        elif center1_x > center2_x + 50:
            h_rel = "right of"
        else:
            h_rel = "aligned with"

        if center1_y < center2_y - 50:
            v_rel = "above"
        elif center1_y > center2_y + 50:
            v_rel = "below"
        else:
            v_rel = "level with"

        return f"{obj1['class']} is {h_rel} and {v_rel} {obj2['class']}"

    def extract_all(self, image_path: str) -> Dict[str, Any]:
        """
        Extract all features from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing all extracted features
        """
        logger.debug(f"Extracting features from {image_path}")

        # Caption-only mode: Only extract caption
        if self.caption_only:
            caption = self.extract_caption(image_path)
            features = {
                'caption': caption,
                'scene': caption,
                'ocr_text': '',
                'objects': [],
                'object_details': [],
                'spatial_relations': [],
                'attributes': []
            }
            return features

        # Full extraction mode: Extract all features
        ocr_text = self.extract_ocr(image_path)
        objects = self.extract_objects(image_path)
        caption = self.extract_caption(image_path)
        spatial_relations = self.extract_spatial_relations(objects)

        # Compile object names
        object_names = list(set([obj['class'] for obj in objects]))

        features = {
            'ocr_text': ocr_text,
            'objects': object_names,
            'object_details': objects,
            'caption': caption,
            'spatial_relations': spatial_relations,
            'scene': caption,  # Use caption as scene description
            'attributes': self._extract_attributes(objects)
        }

        return features

    def _extract_attributes(self, objects: List[Dict[str, Any]]) -> List[str]:
        """Extract attributes from detected objects."""
        attributes = []

        for obj in objects:
            conf = obj['confidence']
            if conf > 0.8:
                attributes.append(f"clear {obj['class']}")
            elif conf > 0.5:
                attributes.append(f"visible {obj['class']}")
            else:
                attributes.append(f"possible {obj['class']}")

        return attributes
