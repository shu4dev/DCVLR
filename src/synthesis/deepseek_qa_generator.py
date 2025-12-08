#!/usr/bin/env python3
"""
Hard QA Pair Generator using DeepSeek API (Multiple Choice Version)
Processes image descriptions from JSONL and generates hard multiple-choice questions with multi-step reasoning.
"""

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
INPUT_JSONL = "/mnt/lustre/koa/scratch/shu4/DCVLR/output/intermediate/stage3_description/all_binned_images_with_descriptions.jsonl"
OUTPUT_DIR = "/mnt/lustre/koa/scratch/shu4/DCVLR/output/intermediate/stage4_qa_generation"
MAX_WORKERS = 10  # Parallel processing threads

# --- HARD QA PROMPT (MODIFIED FOR MCQ) ---
HARD_QA_PROMPT = """
You are constructing HARD training data to teach a small model state-of-the-art style visual reasoning.

You are given:
- A natural language reasoning_description about an image.
- Optional OCR text detected in the image.
- Optional detected_objects from an object detector.

Your goal is to create EXACTLY 2 Multiple Choice Questions (MCQ) that are:
- Non-trivial
- Multi-step
- Based ONLY on the provided information
- Such that the FINAL ANSWER is OBJECTIVELY VERIFIABLE from the provided information

--------------------
INPUT CONTEXT
reasoning_description:
{reasoning_description}

ocr_text (may be empty):
{ocr_text}

detected_objects (may be empty, list of labels or label:count pairs):
{detected_objects}
--------------------

DEFINITION OF "HARD" (TARGET DIFFICULTY)

Treat a difficulty rating of 1–5 where:
1 = trivial, can be answered by copying a phrase
3 = moderate, simple reasoning or one relation
5 = challenging, requires multi-step, compositional reasoning

For this task, EVERY question must have difficulty >= 4.

Hard questions should:
- Require AT LEAST 3 distinct reasoning steps.
- Prefer combining information from multiple sources (reasoning_description, ocr_text, detected_objects).
- Involve one or more of:
  - comparisons (counts, sizes, positions, timings)
  - combinations of conditions ("if…, despite…, because…")
  - causal or hypothetical reasoning ("what is likely", "what would happen if…")
- NOT be answerable by quoting a single span of text.

For EACH question you create:
1. **The Question**: Must be objective and uniquely determined by the context.
2. **The Options**: Provide exactly 4 options (A, B, C, D).
   - One option must be the correct answer.
   - The other three must be "distractors": plausible but incorrect answers based on misinterpretations or partial reasoning.
3. **The Answer**: Must be the single letter of the correct option (A, B, C, or D).

Do NOT rely on external world knowledge, subjective judgments, personal preferences, or unspecified future events.

TASK

1. Carefully read and understand all context.
2. Propose candidate hard questions.
3. For EACH final question:
   - Ensure it meets the hardness criteria above.
   - Construct a clear, step-by-step reasoning_trace with AT LEAST 3 numbered steps.
   - Create 4 options.
   - Select the correct letter.

OUTPUT FORMAT

Return ONLY a single JSON object with this exact structure:

{{
  "samples": [
    {{
      "question": "<hard question 1>",
      "options": [
        "<Option A text>",
        "<Option B text>",
        "<Option C text>",
        "<Option D text>"
      ],
      "reasoning_trace": "<at least 3-step reasoning for question 1>",
      "answer": "<Single Letter: A, B, C, or D>"
    }},
    {{
      "question": "<hard question 2>",
      "options": [
        "<Option A text>",
        "<Option B text>",
        "<Option C text>",
        "<Option D text>"
      ],
      "reasoning_trace": "<at least 3-step reasoning for question 2>",
      "answer": "<Single Letter: A, B, C, or D>"
    }}
  ]
}}

Rules:
- Do not mention the words "reasoning_description", "ocr_text", or "detected_objects" in the questions.
- Do not mention difficulty ratings or that you revised questions.
- Do not include any text outside the JSON object.
"""


def make_api_request(
    reasoning_description: str,
    ocr_text: str,
    detected_objects: list,
    client: OpenAI
) -> dict:
    """
    Make API request to DeepSeek for QA pair generation.
    """
    # Format the prompt with input data
    obj_det_str = json.dumps(detected_objects)
    user_prompt = HARD_QA_PROMPT.format(
        reasoning_description=reasoning_description,
        ocr_text=ocr_text if ocr_text else "",
        detected_objects=obj_det_str
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )

        return response

    except Exception as e:
        raise Exception(f"DeepSeek API error: {str(e)}")


def extract_qa_pairs(response) -> List[Dict[str, Any]]:
    """
    Extract and validate QA pairs from an OpenAI/DeepSeek chat completion response.
    Updated to handle 'options' list and validate 'answer' is a letter.
    """
    try:
        # 1. Extract raw content from the first choice
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Invalid response structure: {e}")

        if not content:
            raise ValueError("Response content is empty")

        content = content.strip()

        # 2. Strip ```json / ``` fences if the model added them
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
                content = "\n".join(lines[1:-1]).strip()

        # 3. Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to recover: find the first '{' or '[' and last '}' or ']'
            start_brace = content.find("{")
            start_bracket = content.find("[")
            if start_brace == -1 and start_bracket == -1:
                raise ValueError(f"Invalid JSON in response (no JSON object/array found): {content[:200]}")

            if start_brace == -1 or (start_bracket != -1 and start_bracket < start_brace):
                start = start_bracket
                end = content.rfind("]")
            else:
                start = start_brace
                end = content.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Invalid JSON in response (could not locate JSON substring): {content[:200]}")

            json_str = content[start:end+1]
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e2:
                raise ValueError(f"Invalid JSON in response even after trimming: {e2}; raw: {content[:200]}")

        # 4. Normalize to a list of samples
        if isinstance(data, dict):
            if "samples" in data:
                samples = data["samples"]
            elif all(k in data for k in ("question", "reasoning_trace", "answer", "options")):
                samples = [data]
            else:
                raise ValueError("JSON object does not contain 'samples' or a valid QA object")
        elif isinstance(data, list):
            samples = data
        else:
            raise ValueError("Top-level JSON must be an object or a list")

        # 5. Validate the samples list
        if not isinstance(samples, list):
            raise ValueError("'samples' must be a list")

        if len(samples) != 2:
            raise ValueError(f"Expected exactly 2 samples, got {len(samples)}")

        valid_answers = {"A", "B", "C", "D"}

        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {idx} is not an object/dict")

            # Check for required fields
            for field in ("question", "reasoning_trace", "answer", "options"):
                if field not in sample:
                    raise ValueError(f"Sample {idx} missing '{field}' field")

            # Validate Question and Reasoning
            if not isinstance(sample["question"], str) or not sample["question"].strip():
                raise ValueError(f"Sample {idx} has empty or non-string 'question'")
            if not isinstance(sample["reasoning_trace"], str) or not sample["reasoning_trace"].strip():
                raise ValueError(f"Sample {idx} has empty or non-string 'reasoning_trace'")

            # Validate Options
            options = sample["options"]
            if not isinstance(options, list) or len(options) != 4:
                raise ValueError(f"Sample {idx} 'options' must be a list of exactly 4 strings")
            if not all(isinstance(opt, str) and opt.strip() for opt in options):
                raise ValueError(f"Sample {idx} contains empty or non-string options")

            # Validate Answer (Must be A, B, C, or D)
            answer = sample["answer"].strip().upper()
            if answer not in valid_answers:
                raise ValueError(f"Sample {idx} answer '{sample['answer']}' is not one of A, B, C, D")
            
            # Normalize answer to uppercase just in case
            sample["answer"] = answer

        return samples

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Unexpected error while extracting QA pairs: {e}")


def process_single_entry(
    entry: dict,
    client: OpenAI,
    index: int
) -> dict:
    """
    Process a single JSONL entry: call API, extract QA pairs.
    """
    try:
        # Validate required field
        reasoning_description = entry.get("reasoning_description")
        if not reasoning_description:
            return {
                "index": index,
                "status": "error",
                "entry": None,
                "error": "Missing 'reasoning_description' field"
            }

        # Get optional fields
        ocr_text = entry.get("ocr_text", "")
        detected_objects = entry.get("detected_objects", [])

        # Make API request
        try:
            response = make_api_request(
                reasoning_description=reasoning_description,
                ocr_text=ocr_text,
                detected_objects=detected_objects,
                client=client
            )
        except Exception as e:
            return {
                "index": index,
                "status": "error",
                "entry": None,
                "error": f"API request failed: {str(e)}"
            }

        # Extract QA pairs
        try:
            qa_pairs = extract_qa_pairs(response)
        except Exception as e:
            return {
                "index": index,
                "status": "error",
                "entry": None,
                "error": f"QA extraction failed: {str(e)}"
            }

        # Enhance entry with QA pairs
        enhanced_entry = dict(entry)  # Shallow copy
        enhanced_entry["qa_pairs"] = qa_pairs

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
    Write enhanced JSONL file with each QA pair as a separate entry.
    Updated to include 'options'.
    """
    success_count = 0
    total_output_entries = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            if result["status"] == "success" and result["entry"] is not None:
                entry = result["entry"]
                qa_pairs = entry.get("qa_pairs", [])

                # Create a separate entry for each QA pair
                for qa_pair in qa_pairs:
                    # Create a copy of the original entry
                    split_entry = dict(entry)

                    # Remove the qa_pairs list and add individual fields
                    split_entry.pop("qa_pairs", None)
                    split_entry["question"] = qa_pair["question"]
                    split_entry["options"] = qa_pair["options"]
                    split_entry["reasoning_trace"] = qa_pair["reasoning_trace"]
                    split_entry["answer"] = qa_pair["answer"]

                    # Write to file
                    json_line = json.dumps(split_entry, ensure_ascii=False)
                    f.write(json_line + "\n")
                    total_output_entries += 1

                success_count += 1

    print(f"\nWrote {total_output_entries} entries ({success_count} original entries × 2) to {output_path}")


def log_errors(error_log_path: str, results: list):
    """
    Log all errors to a separate file for debugging.
    """
    error_count = 0

    with open(error_log_path, 'w', encoding='utf-8') as f:
        f.write("Error Log - QA Pair Generation\n")
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
        description="Hard QA Pair Generator using DeepSeek API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 deepseek_qa_generator.py --api-key sk-xxxxx
  python3 deepseek_qa_generator.py --workers 15
  python3 deepseek_qa_generator.py --input custom_descriptions.jsonl
  python3 deepseek_qa_generator.py --limit 100

API Key:
  The API key can be provided via --api-key argument or DEEPSEEK_API_KEY environment variable.
  The --api-key argument takes precedence if both are provided.
        """
    )
    parser.add_argument(
        "--api-key",
        help="DeepSeek API key (or use DEEPSEEK_API_KEY env var)"
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
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to process (e.g., 100 for testing). Processes all entries if not specified."
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("Error: DeepSeek API key not provided.")
        print("Please provide it via --api-key argument or DEEPSEEK_API_KEY environment variable.")
        return

    print("=" * 80)
    print("Hard QA Pair Generator using DeepSeek API (MCQ Mode)")
    print("=" * 80)

    # Initialize DeepSeek client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = output_dir / "hard_qa_pairs.jsonl"
    error_log = output_dir / "qa_generation_errors.log"

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

    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        original_count = len(entries)
        entries = entries[:args.limit]
        print(f"Limiting to {len(entries)} entries (out of {original_count} total)")

    total = len(entries)
    print(f"\nProcessing {total} entries with {args.workers} workers...")
    print(f"Output will be saved to: {output_jsonl}")
    print(f"Errors will be logged to: {error_log}")
    print()

    # Process entries in parallel
    results = [None] * total

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_entry,
                entry,
                client,
                idx
            ): idx
            for idx, entry in enumerate(entries)
        }

        # Collect results with progress bar
        with tqdm(total=total, desc="Processing entries") as pbar:
            for future in as_completed(futures):
                result = future.result()
                idx = result["index"]
                results[idx] = result

                # Update progress bar description
                if result["status"] == "error":
                    pbar.set_postfix({"last_error": result["error"][:50]})

                pbar.update(1)

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
    print("=" * 80)


if __name__ == "__main__":
    main()