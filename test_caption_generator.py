"""
Test file for Image Caption Generation - standalone usage example.

This script demonstrates how to generate image captions exactly as it's used in the DCVLR repository.
Supports three backends: BLIP, BLIP-2, and Moondream API.
Takes a single image path as input and generates a descriptive caption.
"""

import sys
import logging
from pathlib import Path
from PIL import Image
import torch

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: transformers not installed. Install with: pip install transformers")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CaptionGeneratorTester:
    """Simple wrapper for testing image caption generation."""

    def __init__(self, backend='blip', device='cuda:0', moondream_api_key=None, moondream_length='normal'):
        """
        Initialize caption generator.

        Args:
            backend: Caption backend - 'blip', 'blip2', or 'moondream'
            device: Device to run the model on ('cuda:0', 'cuda:1', or 'cpu')
            moondream_api_key: API key for Moondream (only needed if backend='moondream')
            moondream_length: Caption length for Moondream - 'short', 'normal', or 'long'
        """
        self.captioner_backend = backend.lower()
        self.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'

        if self.device == 'cpu':
            logger.warning("CUDA not available, using CPU (will be slow)")

        # Initialize based on backend
        if self.captioner_backend == 'moondream':
            # Moondream API-based captioning
            if not moondream_api_key:
                logger.error("Moondream API key required for moondream backend")
                raise ValueError("moondream_api_key is required when backend='moondream'")

            logger.info("Loading Moondream API captioner...")
            try:
                from src.synthesis.moondream_captioner import MoondreamCaptioner
                self.moondream_captioner = MoondreamCaptioner(
                    api_key=moondream_api_key,
                    length=moondream_length
                )
                logger.info(f"Moondream API captioner loaded (length={moondream_length})")
                self.blip_model = None
                self.blip_processor = None
            except Exception as e:
                logger.error(f"Failed to load Moondream captioner: {e}")
                raise

        elif self.captioner_backend == 'blip2':
            # BLIP-2 for higher quality captions
            logger.info(f"Loading BLIP-2 on {self.device}...")

            self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

            # Set pad_token if not already set
            if self.blip_processor.tokenizer.pad_token is None:
                self.blip_processor.tokenizer.pad_token = self.blip_processor.tokenizer.eos_token
                self.blip_processor.tokenizer.pad_token_id = self.blip_processor.tokenizer.eos_token_id

            # Determine dtype and device for loading
            model_dtype = torch.float16 if "cuda" in self.device else torch.float32

            # Load model without device_map for better compatibility
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=model_dtype
            )

            # Move model to device explicitly
            if "cuda" in self.device:
                self.blip_model = self.blip_model.to(self.device)

            self.blip_model.eval()

            logger.info(f"BLIP-2 loaded successfully on {self.device}")

        else:  # blip (default)
            # BLIP-base for faster processing
            logger.info(f"Loading BLIP-base on {self.device}...")

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
            self.blip_model.to(self.device)

            logger.info(f"BLIP-base loaded successfully on {self.device}")

    def generate_caption(self, image_path):
        """
        Generate caption for image.

        This method replicates the exact logic from src/filtering/binning.py:638-695

        Args:
            image_path: Path to the image file (str or Path)

        Returns:
            str: Generated caption text
        """
        try:
            logger.info(f"Processing image: {image_path}")

            # Use Moondream API if configured
            if self.captioner_backend == 'moondream':
                caption = self.moondream_captioner.generate_caption(image_path)
                return caption if caption else ""

            # Use BLIP/BLIP-2 for local processing
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Image loaded: {image.size[0]}x{image.size[1]}")

            if self.captioner_backend == 'blip2':
                # BLIP-2 processing (exact logic from binning.py:657-674)
                logger.info("Generating caption with BLIP-2...")

                inputs = self.blip_processor(images=image, return_tensors="pt")

                # Debug input shapes
                logger.debug(f"Input keys: {inputs.keys()}")
                if 'pixel_values' in inputs:
                    logger.debug(f"pixel_values shape: {inputs['pixel_values'].shape}")

                # Move inputs to device with proper dtype
                inputs = inputs.to(
                    self.device,
                    torch.float16 if "cuda" in self.device else torch.float32
                )

                # Check if inputs are valid (not empty)
                if 'pixel_values' in inputs and inputs['pixel_values'].numel() > 0:
                    with torch.no_grad():
                        generated_ids = self.blip_model.generate(
                            **inputs,
                            max_length=50,
                            pad_token_id=self.blip_processor.tokenizer.pad_token_id
                        )
                    caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                else:
                    logger.warning(f"Empty input tensors for {image_path}, skipping caption generation")
                    return ""
            else:
                # BLIP-base processing (exact logic from binning.py:676-690)
                logger.info("Generating caption with BLIP-base...")

                inputs = self.blip_processor(image, return_tensors="pt").to(self.device)

                # Check if inputs are valid (not empty)
                if 'pixel_values' in inputs and inputs['pixel_values'].numel() > 0:
                    with torch.no_grad():
                        out = self.blip_model.generate(
                            **inputs,
                            max_new_tokens=50,
                            pad_token_id=self.blip_processor.tokenizer.pad_token_id
                        )
                    caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                else:
                    logger.warning(f"Empty input tensors for {image_path}, skipping caption generation")
                    return ""

            logger.info(f"Caption generated successfully")
            return caption

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return ""


def main():
    """Main function to test caption generation on a single image."""
    import argparse

    parser = argparse.ArgumentParser(description='Test image caption generation')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--backend', type=str, default='blip2',
                        choices=['blip', 'blip2', 'moondream'],
                        help='Caption backend (default: blip)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on (default: cuda:0)')
    parser.add_argument('--moondream-api-key', type=str, default=None,
                        help='Moondream API key (required if backend=moondream)')
    parser.add_argument('--moondream-length', type=str, default='normal',
                        choices=['short', 'normal', 'long'],
                        help='Moondream caption length (default: normal)')

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"ERROR: Image not found: {args.image_path}")
        sys.exit(1)

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = args.device
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")

    # Validate moondream requirements
    if args.backend == 'moondream' and not args.moondream_api_key:
        print("ERROR: --moondream-api-key is required when using moondream backend")
        sys.exit(1)

    # Initialize tester
    try:
        tester = CaptionGeneratorTester(
            backend=args.backend,
            device=device,
            moondream_api_key=args.moondream_api_key,
            moondream_length=args.moondream_length
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize caption generator: {e}")
        sys.exit(1)

    # Generate caption
    caption = tester.generate_caption(args.image_path)

    # Print results
    print("\n" + "="*60)
    print("IMAGE CAPTION GENERATION RESULTS")
    print("="*60)
    print(f"Image: {args.image_path}")
    print(f"Backend: {args.backend.upper()}")
    print(f"Device: {device}")
    print(f"\nGenerated Caption:")
    print(f"  {caption}")
    print("="*60)


if __name__ == "__main__":
    main()
