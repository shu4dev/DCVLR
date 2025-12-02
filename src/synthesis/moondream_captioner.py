"""
Moondream API captioning module for the Team-1 Data Synthesis Pipeline.
Provides image captioning using the Moondream API.
"""

import logging
import base64
import requests
from typing import Optional
from pathlib import Path
from PIL import Image
import io

logger = logging.getLogger(__name__)


class MoondreamCaptioner:
    """
    Generates image captions using the Moondream API.

    Usage:
        captioner = MoondreamCaptioner(api_key="your_api_key")
        caption = captioner.generate_caption("path/to/image.jpg")
    """

    API_ENDPOINT = "https://api.moondream.ai/v1/caption"

    def __init__(
        self,
        api_key: str,
        length: str = "long",
        stream: bool = False,
        timeout: int = 30
    ):
        """
        Initialize the Moondream captioner.

        Args:
            api_key: Moondream API key (required)
            length: Caption length preference ("short", "normal", or "long")
            stream: Whether to use streaming responses
            timeout: Request timeout in seconds
        """
        if not api_key:
            raise ValueError("Moondream API key is required")

        self.api_key = api_key
        self.length = length
        self.stream = stream
        self.timeout = timeout

        logger.info(f"MoondreamCaptioner initialized with length='{length}'")

    def _image_to_base64_uri(self, image_path: str) -> str:
        """
        Convert an image file to a base64 data URI.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded data URI string
        """
        try:
            # Open and convert image to RGB
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Save to bytes buffer as JPEG
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)

            # Encode to base64
            image_bytes = buffer.read()
            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')

            # Create data URI
            data_uri = f"data:image/jpeg;base64,{base64_encoded}"

            return data_uri

        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise

    def generate_caption(self, image_path: str) -> Optional[str]:
        """
        Generate a caption for an image using the Moondream API.

        Args:
            image_path: Path to the image file

        Returns:
            Generated caption string, or None if request fails
        """
        try:
            # Convert image to base64 data URI
            image_uri = self._image_to_base64_uri(image_path)

            # Prepare request
            headers = {
                'Content-Type': 'application/json',
                'X-Moondream-Auth': self.api_key
            }

            payload = {
                'image_url': image_uri,
                'length': self.length,
                'stream': self.stream
            }

            # Make API request
            logger.debug(f"Requesting caption for {Path(image_path).name}")
            response = requests.post(
                self.API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            # Check response
            response.raise_for_status()

            # Parse response
            result = response.json()
            caption = result.get('caption', '')

            # Log metrics if available
            if 'metrics' in result:
                metrics = result['metrics']
                logger.debug(
                    f"Caption generated: {len(caption)} chars, "
                    f"tokens: {metrics.get('input_tokens', 0)}→{metrics.get('output_tokens', 0)}"
                )

            return caption

        except requests.exceptions.RequestException as e:
            logger.error(f"Moondream API request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            return None

        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return None

    def generate_captions_batch(self, image_paths: list) -> list:
        """
        Generate captions for multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of generated captions (None for failed requests)
        """
        captions = []

        for i, img_path in enumerate(image_paths, 1):
            logger.info(f"Processing image #{i}/{len(image_paths)}: {Path(img_path).name}")
            caption = self.generate_caption(img_path)
            captions.append(caption)

        logger.info(f"Generated {sum(1 for c in captions if c)} captions out of {len(image_paths)}")
        return captions


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check for API key
    if len(sys.argv) < 3:
        print("Usage: python moondream_captioner.py <api_key> <image_path>")
        print("Example: python moondream_captioner.py YOUR_API_KEY image.jpg")
        sys.exit(1)

    api_key = sys.argv[1]
    image_path = sys.argv[2]

    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    # Test the captioner
    print(f"\nGenerating caption for: {image_path}")
    print("-" * 60)

    captioner = MoondreamCaptioner(api_key=api_key, length="normal")
    caption = captioner.generate_caption(image_path)

    if caption:
        print(f"\n✓ Caption: {caption}\n")
    else:
        print("\n✗ Caption generation failed\n")
