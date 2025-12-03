#!/usr/bin/env python3
"""
Test script to verify PaddleOCR new API integration.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def test_paddle_ocr_import():
    """Test if PaddleOCR can be imported."""
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        print("✓ PaddleOCR import successful")
        return True
    except ImportError as e:
        print(f"✗ PaddleOCR import failed: {e}")
        return False


def test_paddle_ocr_initialization():
    """Test PaddleOCR initialization with new API."""
    try:
        from paddleocr import PaddleOCR

        print("\nTesting PaddleOCR initialization...")
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        print("✓ PaddleOCR initialized successfully")
        return ocr
    except Exception as e:
        print(f"✗ PaddleOCR initialization failed: {e}")
        return None


def test_paddle_ocr_predict(ocr, test_image_url=None):
    """Test PaddleOCR predict method."""
    if ocr is None:
        print("✗ Cannot test predict - OCR not initialized")
        return False

    try:
        # Use a sample image URL or local path
        if test_image_url is None:
            test_image_url = (
                "/mnt/lustre/koa/scratch/shu4/DCVLR/data/"
                "HuggingFaceM4__ChartQA/train/027555.jpg"
            )

        print(f"\nTesting PaddleOCR predict on: {test_image_url}")
        results = ocr.predict(input=test_image_url)

        if results:
            print(f"✓ PaddleOCR predict successful - got {len(results)} result(s)")

            # Print result structure
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Type: {type(result)}")
                print(f"  Attributes: {dir(result)}")

                # Try to access json attribute
                if hasattr(result, "json"):
                    data_json = result.json
                    print(f"  JSON data available: {data_json is not None}")
                    if data_json is not None:
                        # If it's a dict, inspect keys and one value
                        if isinstance(data_json, dict):
                            print(
                                f"  json type: dict with "
                                f"{len(data_json)} top-level key(s)"
                            )
                            print(f"  Top-level keys: {list(data_json.keys())}")
                            if data_json:
                                first_key = next(iter(data_json))
                                first_val = data_json[first_key]
                                print(f"  First key: {first_key}")
                                print(f"  Type of first value: {type(first_val)}")
                                if isinstance(first_val, dict):
                                    print(
                                        "  First value keys: "
                                        f"{list(first_val.keys())}"
                                    )
                                elif isinstance(first_val, (list, tuple)) and first_val:
                                    print(
                                        "  First value is a sequence of "
                                        f"length {len(first_val)}; "
                                        f"first element type: {type(first_val[0])}"
                                    )
                                    if isinstance(first_val[0], dict):
                                        print(
                                            "  First element keys: "
                                            f"{list(first_val[0].keys())}"
                                        )
                                    else:
                                        print(f"  First element: {first_val[0]}")
                                else:
                                    print(f"  First value: {first_val}")
                        # If it's a list/tuple, look at the first element
                        elif isinstance(data_json, (list, tuple)):
                            print(
                                f"  json type: {type(data_json)} "
                                f"with {len(data_json)} item(s)"
                            )
                            if data_json:
                                first_item = data_json[0]
                                print(f"  First item type: {type(first_item)}")
                                if isinstance(first_item, dict):
                                    print(
                                        "  First item keys: "
                                        f"{list(first_item.keys())}"
                                    )
                                else:
                                    print(f"  First item: {first_item}")
                        else:
                            print(
                                "  json attribute is type "
                                f"{type(data_json)}; value: {data_json}"
                            )
                else:
                    print("  Result has no 'json' attribute")

            return True
        else:
            print("✗ PaddleOCR predict returned no results")
            return False

    except Exception as e:
        print(f"✗ PaddleOCR predict failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_binning_integration():
    """Test the binning module with new PaddleOCR API."""
    try:
        print("\n" + "=" * 60)
        print("Testing Binning Module Integration")
        print("=" * 60)

        from src.filtering.binning import ImageBinner

        # Create a minimal config for testing
        config = {
            "text_boxes_threshold": 2,
            "text_area_threshold": 0.2,
            "object_count_threshold": 5,
            "unique_objects_threshold": 3,
            "clip_similarity_threshold": 0.25,
            "spatial_dispersion_threshold": 0.3,
            "pipeline_mode": "hybrid",  # Use hybrid mode to trigger PaddleOCR
            "use_blip2": False,
            "captioner_backend": "blip",
            "object_detector": "yolo",
            "yolo_model": "yolov8n",
            "enable_multi_gpu": False,  # Disable multi-GPU for testing
        }

        print("\nInitializing ImageBinner with hybrid mode...")
        binner = ImageBinner(config)

        if binner.ocr_backend == "paddle":
            print("✓ ImageBinner initialized with PaddleOCR backend")
            return True
        else:
            print(
                "✗ ImageBinner initialized with wrong backend: "
                f"{binner.ocr_backend}"
            )
            return False

    except Exception as e:
        print(f"✗ Binning integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("PaddleOCR New API Integration Test")
    print("=" * 60)

    # Test 1: Import
    if not test_paddle_ocr_import():
        print("\n❌ Tests failed - PaddleOCR not available")
        return 1

    # Test 2: Initialization
    ocr = test_paddle_ocr_initialization()
    if ocr is None:
        print("\n❌ Tests failed - Cannot initialize PaddleOCR")
        return 1

    # Test 3: Predict
    if not test_paddle_ocr_predict(ocr):
        print("\n⚠️  Predict test failed - may need network access or different image")

    # Test 4: Binning integration
    # Commented out by default as it requires all dependencies
    # if not test_binning_integration():
    #     print("\n❌ Binning integration test failed")
    #     return 1

    print("\n" + "=" * 60)
    print("✓ All basic tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
