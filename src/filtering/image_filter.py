"""
Image filtering module for the Team-1 Data Synthesis Pipeline.
Handles resolution, NSFW, watermark detection, and deduplication.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

import numpy as np
from PIL import Image
import imagehash
from nsfw_detector import NSFWDetector
import cv2

logger = logging.getLogger(__name__)


class ImageFilter:
    """
    Handles image filtering operations including:
    - Resolution checks
    - NSFW content detection
    - Watermark detection
    - Duplicate detection
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the image filter with configuration.
        
        Args:
            config: Dictionary containing filter configuration
        """
        self.config = config
        self.min_resolution = config.get('min_resolution', 256)
        self.nsfw_threshold = config.get('nsfw_threshold', 0.5)
        self.phash_threshold = config.get('phash_threshold', 8)
        
        # Initialize NSFW detector
        try:
            self.nsfw_detector = NSFWDetector()
        except Exception as e:
            logger.warning(f"Could not initialize NSFW detector: {e}")
            self.nsfw_detector = None
        
        # Store hashes for duplicate detection
        self.seen_hashes: Set[imagehash.ImageHash] = set()
    
    def check_resolution(self, image_path: str) -> bool:
        """
        Check if image meets minimum resolution requirements.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image meets resolution requirements
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                return (width >= self.min_resolution and 
                       height >= self.min_resolution)
        except Exception as e:
            logger.error(f"Error checking resolution for {image_path}: {e}")
            return False
    
    def check_nsfw(self, image_path: str) -> bool:
        """
        Check if image contains NSFW content.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image is safe (not NSFW)
        """
        if not self.nsfw_detector:
            # If detector not available, assume safe
            return True
        
        try:
            with Image.open(image_path) as img:
                is_nsfw = self.nsfw_detector.is_nsfw(img)
                return not is_nsfw
        except Exception as e:
            logger.warning(f"Error checking NSFW for {image_path}: {e}")
            # In case of error, be conservative and filter out
            return False
    
    def check_watermark(self, image_path: str) -> bool:
        """
        Check if image contains watermarks.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if no significant watermarks detected
        """
        try:
            # Simple watermark detection using edge detection
            img = cv2.imread(str(image_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Look for text in corners (common watermark locations)
            h, w = gray.shape
            corner_size = min(h, w) // 4
            
            corners = [
                gray[:corner_size, :corner_size],  # Top-left
                gray[:corner_size, -corner_size:],  # Top-right
                gray[-corner_size:, :corner_size],  # Bottom-left
                gray[-corner_size:, -corner_size:]  # Bottom-right
            ]
            
            for corner in corners:
                # Apply edge detection
                edges = cv2.Canny(corner, 50, 150)
                edge_ratio = np.sum(edges > 0) / edges.size
                
                # If too many edges in corner, likely a watermark
                if edge_ratio > 0.3:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking watermark for {image_path}: {e}")
            # Assume no watermark if check fails
            return True
    
    def is_duplicate(self, image_path: str) -> bool:
        """
        Check if image is a duplicate using perceptual hashing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image is a duplicate
        """
        try:
            with Image.open(image_path) as img:
                # Calculate perceptual hash
                phash = imagehash.phash(img)
                
                # Check against seen hashes
                for seen_hash in self.seen_hashes:
                    if phash - seen_hash < self.phash_threshold:
                        return True
                
                # Add to seen hashes
                self.seen_hashes.add(phash)
                return False
                
        except Exception as e:
            logger.error(f"Error checking duplicate for {image_path}: {e}")
            return False
    
    def filter_images(self, image_paths: List[str]) -> List[str]:
        """
        Apply all filters to a list of image paths.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of paths that passed all filters
        """
        filtered = []
        
        for path in image_paths:
            # Apply all filters
            if not self.check_resolution(path):
                logger.debug(f"Filtered {path}: low resolution")
                continue
                
            if not self.check_nsfw(path):
                logger.debug(f"Filtered {path}: NSFW content")
                continue
                
            if not self.check_watermark(path):
                logger.debug(f"Filtered {path}: watermark detected")
                continue
                
            if self.is_duplicate(path):
                logger.debug(f"Filtered {path}: duplicate")
                continue
            
            filtered.append(path)
        
        logger.info(f"Filtered {len(image_paths)} images to {len(filtered)}")
        return filtered
    
    def reset_duplicates(self):
        """Reset the duplicate detection cache."""
        self.seen_hashes.clear()


class DuplicateDetector:
    """
    Advanced duplicate detection using multiple methods.
    """
    
    def __init__(self, threshold: int = 8):
        """
        Initialize duplicate detector.
        
        Args:
            threshold: Hamming distance threshold for duplicates
        """
        self.threshold = threshold
        self.hashes = {
            'phash': {},
            'ahash': {},
            'dhash': {},
            'whash': {}
        }
    
    def add_image(self, image_path: str, image_id: str):
        """
        Add an image to the duplicate detector.
        
        Args:
            image_path: Path to the image file
            image_id: Unique identifier for the image
        """
        try:
            with Image.open(image_path) as img:
                self.hashes['phash'][image_id] = imagehash.phash(img)
                self.hashes['ahash'][image_id] = imagehash.average_hash(img)
                self.hashes['dhash'][image_id] = imagehash.dhash(img)
                self.hashes['whash'][image_id] = imagehash.whash(img)
        except Exception as e:
            logger.error(f"Error adding image {image_path}: {e}")
    
    def find_duplicates(self, image_path: str) -> List[str]:
        """
        Find duplicate images for the given image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of image IDs that are duplicates
        """
        duplicates = []
        
        try:
            with Image.open(image_path) as img:
                # Calculate hashes for the image
                phash = imagehash.phash(img)
                ahash = imagehash.average_hash(img)
                dhash = imagehash.dhash(img)
                whash = imagehash.whash(img)
                
                # Check against stored hashes
                for image_id in self.hashes['phash']:
                    # Use multiple hash types for robust detection
                    phash_dist = phash - self.hashes['phash'][image_id]
                    ahash_dist = ahash - self.hashes['ahash'][image_id]
                    dhash_dist = dhash - self.hashes['dhash'][image_id]
                    whash_dist = whash - self.hashes['whash'][image_id]
                    
                    # If any hash is very similar, consider duplicate
                    if (phash_dist < self.threshold or 
                        ahash_dist < self.threshold or 
                        dhash_dist < self.threshold or 
                        whash_dist < self.threshold):
                        duplicates.append(image_id)
                
        except Exception as e:
            logger.error(f"Error finding duplicates for {image_path}: {e}")
        
        return duplicates
    
    def remove_duplicates_batch(self, image_data: List[Dict]) -> List[Dict]:
        """
        Remove duplicates from a batch of images.
        
        Args:
            image_data: List of image dictionaries with 'path' and 'id' keys
            
        Returns:
            List of unique images
        """
        unique_images = []
        
        for img_data in image_data:
            duplicates = self.find_duplicates(img_data['path'])
            
            if not duplicates:
                # No duplicates found, add to unique list
                unique_images.append(img_data)
                self.add_image(img_data['path'], img_data['id'])
        
        logger.info(f"Removed {len(image_data) - len(unique_images)} duplicates")
        return unique_images
