"""
Test file for DeepSeek OCR - standalone usage example.

This script demonstrates how to use DeepSeek-OCR exactly as it's used in the DCVLR repository.
Takes a single image path as input and performs OCR text detection.
"""

import sys
import logging
import re
from pathlib import Path
from PIL import Image
import torch

try:
    from transformers import AutoModel, AutoTokenizer
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    print("ERROR: transformers not installed. Install with: pip install transformers")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepSeekOCRTester:
    """Simple wrapper for testing DeepSeek-OCR on a single image."""

    def __init__(self, device='cuda:0', model_size='tiny'):
        """
        Initialize DeepSeek-OCR model.

        Args:
            device: Device to run the model on ('cuda:0', 'cuda:1', or 'cpu')
            model_size: Model size - 'tiny', 'small', 'base', 'large', or 'gundam'
        """
        self.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'

        if self.device == 'cpu':
            logger.warning("CUDA not available, using CPU (will be slow)")

        logger.info(f"Loading DeepSeek-OCR on {self.device}...")

        # Load model and tokenizer from HuggingFace
        model_name = 'deepseek-ai/DeepSeek-OCR'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Determine dtype based on device
        model_dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        self.model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation='flash_attention_2',
            torch_dtype=model_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
            use_safetensors=True
        )

        # Ensure model is on correct device if device_map didn't work
        if self.device != "cpu" and self.model.device.type == 'cpu':
            self.model = self.model.to(self.device)

        self.model = self.model.eval()

        # Set model size parameters based on config
        model_size = model_size.lower()
        if model_size == 'tiny':
            self.base_size = 512
            self.image_size = 512
            self.crop_mode = False
        elif model_size == 'small':
            self.base_size = 640
            self.image_size = 640
            self.crop_mode = False
        elif model_size == 'base':
            self.base_size = 1024
            self.image_size = 1024
            self.crop_mode = False
        elif model_size == 'large':
            self.base_size = 1280
            self.image_size = 1280
            self.crop_mode = False
        else:  # gundam (default)
            self.base_size = 1024
            self.image_size = 640
            self.crop_mode = True

        logger.info(f"DeepSeek-OCR loaded successfully (size: {model_size})")
        logger.info(f"Model parameters: base_size={self.base_size}, image_size={self.image_size}, crop_mode={self.crop_mode}")

    def detect_text(self, image_path):
        """
        Detect text in image using DeepSeek-OCR.

        This method replicates the exact logic from src/filtering/binning.py:394-491

        Args:
            image_path: Path to the image file (str or Path)

        Returns:
            dict: Dictionary containing:
                - num_boxes: Number of text boxes detected
                - text_area_ratio: Ratio of text area to total image area
                - bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
                - raw_result: Raw result string from DeepSeek-OCR
        """
        try:
            # Get image dimensions
            img = Image.open(image_path)
            img_w, img_h = img.size
            img_area = img_w * img_h

            logger.info(f"Processing image: {image_path}")
            logger.info(f"Image size: {img_w}x{img_h} ({img_area} pixels)")

            # Use grounding mode to detect text regions
            # This is the exact prompt used in binning.py:428
            prompt = "<image>\n<|grounding|>Convert the document to markdown."

            # Call the infer method with proper parameters
            logger.info("Running DeepSeek-OCR inference...")

            # Create a temporary directory for output files (DeepSeek-OCR requires this)
            import tempfile
            import os
            import shutil
            temp_dir = tempfile.mkdtemp(prefix='deepseek_ocr_')

            try:
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=temp_dir,  # Use temp directory
                    base_size=self.base_size,
                    image_size=self.image_size,
                    crop_mode=self.crop_mode,
                    save_results=False,
                    test_compress=False
                )
            except Exception as ocr_error:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.error(f"DeepSeek OCR failed: {ocr_error}")
                return {
                    'num_boxes': 0,
                    'text_area_ratio': 0.0,
                    'bboxes': [],
                    'raw_result': None,
                    'error': str(ocr_error)
                }
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)

            logger.info("OCR inference complete")

            bboxes = []

            # Parse the result text to extract bounding boxes
            # DeepSeek OCR with grounding mode returns text with coordinates
            if result and isinstance(result, str):
                logger.info(f"Raw OCR result length: {len(result)} characters")

                # Pattern to match coordinates like [x1,y1,x2,y2]
                # This is the exact pattern from binning.py:452
                pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
                matches = re.findall(pattern, result)

                logger.info(f"Found {len(matches)} bounding box matches")

                if matches:
                    # Filter out invalid bboxes (e.g., [0,0,999,999] which covers entire image)
                    # This is the exact logic from binning.py:456-460
                    valid_matches = []
                    for x1, y1, x2, y2 in matches:
                        # Skip bboxes that cover the entire normalized space
                        if not (int(x1) == 0 and int(y1) == 0 and int(x2) == 999 and int(y2) == 999):
                            valid_matches.append((x1, y1, x2, y2))

                    logger.info(f"Valid bounding boxes: {len(valid_matches)}")

                    if valid_matches:
                        # Convert normalized coordinates (0-999) to actual pixels
                        # This is the exact logic from binning.py:464-468
                        bboxes = [
                            [int(x1) * img_w / 1000, int(y1) * img_h / 1000,
                             int(x2) * img_w / 1000, int(y2) * img_h / 1000]
                            for x1, y1, x2, y2 in valid_matches
                        ]

            if not bboxes:
                logger.info("No text boxes detected")
                return {
                    'num_boxes': 0,
                    'text_area_ratio': 0.0,
                    'bboxes': [],
                    'raw_result': result
                }

            # Count text boxes
            num_boxes = len(bboxes)

            # Calculate text area ratio
            # This is the exact logic from binning.py:477-485
            text_area = 0
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                box_w = abs(x2 - x1)
                box_h = abs(y2 - y1)
                text_area += box_w * box_h

            # Avoid division by zero
            text_area_ratio = text_area / img_area if img_area > 0 else 0.0

            logger.info(f"Results: {num_boxes} text boxes, {text_area_ratio:.2%} text area ratio")

            return {
                'num_boxes': num_boxes,
                'text_area_ratio': text_area_ratio,
                'bboxes': bboxes,
                'raw_result': result
            }

        except Exception as e:
            logger.error(f"Error detecting text: {e}")
            return {
                'num_boxes': 0,
                'text_area_ratio': 0.0,
                'bboxes': [],
                'raw_result': None,
                'error': str(e)
            }


def main():
    """Main function to test DeepSeek-OCR on a single image."""
    if len(sys.argv) != 2:
        print("Usage: python test_deepseek_ocr.py <image_path>")
        print("\nExample:")
        print("  python test_deepseek_ocr.py /path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # Check if image exists
    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")

    # Initialize tester (using 'tiny' model size for memory efficiency)
    # You can change this to 'small', 'base', 'large', or 'gundam'
    tester = DeepSeekOCRTester(device=device, model_size='tiny')

    # Detect text
    result = tester.detect_text(image_path)

    # Print results
    print("\n" + "="*60)
    print("DEEPSEEK-OCR TEST RESULTS")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Number of text boxes: {result['num_boxes']}")
    print(f"Text area ratio: {result['text_area_ratio']:.2%}")

    if result['bboxes']:
        print(f"\nBounding boxes (pixel coordinates):")
        for i, bbox in enumerate(result['bboxes'], 1):
            x1, y1, x2, y2 = bbox
            print(f"  Box {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    if result.get('error'):
        print(f"\nError: {result['error']}")

    if result['raw_result'] and len(result['raw_result']) < 1000:
        print(f"\nRaw OCR output:")
        print(result['raw_result'])
    elif result['raw_result']:
        print(f"\nRaw OCR output (truncated, {len(result['raw_result'])} chars):")
        print(result['raw_result'][:500] + "...")

    print("="*60)


if __name__ == "__main__":
    main()
