#!/usr/bin/env python3
"""
Batch Image Description Generator
Processes all images from JSONL file using OpenRouter API with parallel processing.
"""

import requests
import json
import base64
import time
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_JSONL = "/mnt/lustre/koa/scratch/shu4/DCVLR/output/intermediate/stage2_binning/all_binned_images.jsonl"
OUTPUT_DIR = "/mnt/lustre/koa/scratch/shu4/DCVLR/output/intermediate/stage3_description"
MAX_WORKERS = 10  # Parallel processing threads
MAX_RETRIES = 5   # API retry attempts
BASE_DELAY = 1.0  # Base delay for exponential backoff (seconds)

# --- SYSTEM PROMPT ---
SYSTEM_INSTRUCTION = """
# Role
You are an expert Computer Vision Analyst specializing in granular image description for the creation of multimodal reasoning datasets.

# Instructions
Generate a structured description of the image. Synthesize the visual features with the provided OCR and Object Detection data.
Your output must adhere to the following structure:

## 1. High-Level Summary
Provide a concise 2-sentence overview of the scene.

## 2. Detailed Visual Taxonomy
Describe specific elements: Main Subjects, Object Attributes, State of Action.

## 3. Spatial Topology & Layout
Describe geometry: Relative Positioning (left/right/foreground), Depth, and Occlusion.

## 4. Text & Semantic Integration (OCR Context)
Locate provided OCR text within the scene. Explicitly map text to objects.

## 5. Causal & Contextual Inferences
Describe details that support logical reasoning: Lighting, implied events, relationships.

# Constraints
* Be descriptive but objective.
* If OCR/Object data contradicts visual evidence, prioritize visual evidence but note the discrepancy.
* Aim for high information density.
"""


def encode_image(path: str) -> str:
    """Encode image to base64 with error handling."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {path}")
    except Exception as e:
        raise IOError(f"Failed to read image {path}: {str(e)}")


def make_api_request(
    b64_image: str,
    ocr_text: str,
    detected_objects: list,
    api_key: str,
    system_instruction: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> dict:
    """
    Make API request with exponential backoff retry logic.

    Returns:
        dict: API response with 'choices' containing description

    Raises:
        Exception: After max_retries exhausted
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    obj_det_str = json.dumps(detected_objects)
    payload = {
        "model": "google/gemini-2.0-flash-exp:free",
        "messages": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"**OCR Data:** {ocr_text}\n**Object Detection Data:** {obj_det_str}\n\nPlease generate the description."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    }
                ]
            }
        ]
    }

    last_exception = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

            # Check HTTP status
            if response.status_code == 429:  # Rate limit
                wait_time = base_delay * (2 ** attempt) * 2
                time.sleep(wait_time)
                continue
            elif response.status_code >= 500:  # Server error
                wait_time = base_delay * (2 ** attempt)
                time.sleep(wait_time)
                continue
            elif response.status_code != 200:
                # Client error - no retry
                raise Exception(f"API error {response.status_code}: {response.text}")

            # Parse response
            response_data = response.json()

            # Check for API-level errors
            if 'error' in response_data:
                raise Exception(f"API error: {response_data['error']}")

            # Rate limiting delay (even on success)
            time.sleep(0.5)

            return response_data

        except requests.exceptions.Timeout:
            last_exception = Exception(f"Request timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
        except requests.exceptions.RequestException as e:
            last_exception = Exception(f"Network error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
        except json.JSONDecodeError as e:
            last_exception = Exception(f"Invalid JSON response: {str(e)}")
            break  # Don't retry on parse errors
        except Exception as e:
            last_exception = e
            if "API error" in str(e):
                break  # Don't retry on client errors
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))

    # All retries exhausted
    raise last_exception or Exception("Request failed after all retries")


def extract_description(response_data: dict) -> str:
    """
    Extract description from API response.

    Args:
        response_data: Full API response dict

    Returns:
        str: The generated description text

    Raises:
        ValueError: If response structure is invalid
    """
    if 'choices' not in response_data:
        raise ValueError("Response missing 'choices' field")

    if not response_data['choices']:
        raise ValueError("Response 'choices' array is empty")

    message = response_data['choices'][0].get('message', {})
    content = message.get('content', '')

    if not content:
        raise ValueError("Response content is empty")

    return content.strip()


def process_single_image(
    entry: dict,
    api_key: str,
    system_instruction: str,
    index: int
) -> dict:
    """
    Process a single image: encode, call API, extract description.

    Args:
        entry: JSONL entry containing path, ocr_text, detected_objects, etc.
        api_key: OpenRouter API key
        system_instruction: System prompt for API
        index: Entry index (for tracking)

    Returns:
        dict: {
            "index": int,
            "status": "success" | "error",
            "entry": dict (enhanced with reasoning_description) | None,
            "error": str | None
        }
    """
    try:
        # Validate required fields
        img_path = entry.get("path")
        if not img_path:
            return {
                "index": index,
                "status": "error",
                "entry": None,
                "error": "Missing 'path' field"
            }

        ocr_text = entry.get("ocr_text", "")
        detected_objects = entry.get("detected_objects", [])

        # Encode image
        try:
            b64_image = encode_image(img_path)
        except Exception as e:
            return {
                "index": index,
                "status": "error",
                "entry": None,
                "error": f"Image encoding failed: {str(e)}"
            }

        # Make API request
        try:
            response_data = make_api_request(
                b64_image=b64_image,
                ocr_text=ocr_text,
                detected_objects=detected_objects,
                api_key=api_key,
                system_instruction=system_instruction
            )
        except Exception as e:
            return {
                "index": index,
                "status": "error",
                "entry": None,
                "error": f"API request failed: {str(e)}"
            }

        # Extract description
        try:
            description = extract_description(response_data)
        except Exception as e:
            return {
                "index": index,
                "status": "error",
                "entry": None,
                "error": f"Description extraction failed: {str(e)}"
            }

        # Enhance entry with new field
        enhanced_entry = dict(entry)  # Shallow copy
        enhanced_entry["reasoning_description"] = description

        return {
            "index": index,
            "status": "success",
            "entry": enhanced_entry,
            "error": None
        }

    except Exception as e:
        # Catch-all for unexpected errors
        return {
            "index": index,
            "status": "error",
            "entry": None,
            "error": f"Unexpected error: {str(e)}"
        }


def load_jsonl_entries(jsonl_path: str) -> list:
    """
    Load JSONL file with validation and error handling.

    Args:
        jsonl_path: Path to input JSONL file

    Returns:
        list: List of entry dicts

    Raises:
        FileNotFoundError: If JSONL file doesn't exist
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    entries = []
    skipped_lines = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                skipped_lines += 1

    if skipped_lines > 0:
        print(f"Loaded {len(entries)} entries, skipped {skipped_lines} invalid lines")
    else:
        print(f"Loaded {len(entries)} entries")

    return entries


def write_enhanced_jsonl(output_path: str, entries: list, results: list):
    """
    Write enhanced JSONL file with reasoning_description field.

    Args:
        output_path: Path to output JSONL file
        entries: Original entries list
        results: Processing results list (must match entries by index)
    """
    success_count = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            if result["status"] == "success" and result["entry"] is not None:
                json_line = json.dumps(result["entry"], ensure_ascii=False)
                f.write(json_line + "\n")
                success_count += 1

    print(f"\nWrote {success_count} enhanced entries to {output_path}")


def log_errors(error_log_path: str, results: list):
    """
    Log all errors to a separate file for debugging.

    Args:
        error_log_path: Path to error log file
        results: Processing results list
    """
    error_count = 0

    with open(error_log_path, 'w', encoding='utf-8') as f:
        f.write("Error Log - Image Processing\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            if result["status"] == "error":
                error_count += 1
                f.write(f"Index: {result['index']}\n")
                f.write(f"Error: {result['error']}\n")
                f.write("-" * 80 + "\n")

    if error_count > 0:
        print(f"\nLogged {error_count} errors to {error_log_path}")


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Batch Image Description Generator using OpenRouter API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 req.py --api-key sk-or-v1-xxxxx
  python3 req.py --api-key sk-or-v1-xxxxx --workers 15
  python3 req.py --api-key sk-or-v1-xxxxx --input custom_images.jsonl
        """
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenRouter API key (required)"
    )
    parser.add_argument(
        "--input",
        default=INPUT_JSONL,
        help=f"Path to input JSONL file (default: {INPUT_JSONL})"
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory for results (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Maximum API retry attempts (default: {MAX_RETRIES})"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Batch Image Description Generator")
    print("=" * 80)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = output_dir / "all_binned_images_with_descriptions.jsonl"
    error_log = output_dir / "processing_errors.log"

    # Load input JSONL
    print(f"\nLoading entries from: {args.input}")
    try:
        entries = load_jsonl_entries(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not entries:
        print("No entries to process. Exiting.")
        return

    total = len(entries)
    print(f"\nProcessing {total} images with {args.workers} workers...")
    print(f"Output will be saved to: {output_jsonl}")
    print(f"Errors will be logged to: {error_log}")
    print()

    # Process images in parallel
    results = [None] * total
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_image,
                entry,
                args.api_key,
                SYSTEM_INSTRUCTION,
                idx
            ): idx
            for idx, entry in enumerate(entries)
        }

        # Collect results with progress bar
        with tqdm(total=total, desc="Processing images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                idx = result["index"]
                results[idx] = result

                # Update progress bar description
                if result["status"] == "error":
                    pbar.set_postfix({"last_error": result["error"][:50]})

                pbar.update(1)

    elapsed_time = time.time() - start_time

    # Write outputs
    print("\nWriting results...")
    write_enhanced_jsonl(output_jsonl, entries, results)
    log_errors(error_log, results)

    # Generate statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    print("\n" + "=" * 80)
    print("Processing Complete")
    print("=" * 80)
    print(f"Total entries: {total}")
    print(f"Successful: {success_count} ({success_count/total*100:.1f}%)")
    print(f"Failed: {error_count} ({error_count/total*100:.1f}%)")
    print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"Average time per image: {elapsed_time/total:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
