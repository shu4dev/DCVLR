#!/usr/bin/env python3
"""
Retry Failed Image Descriptions
Identifies and retries entries that failed during initial API processing.

Usage:
    python3 retry_failed_descriptions.py identify --input INPUT --output OUTPUT --save-failed FAILED
    python3 retry_failed_descriptions.py retry --api-key KEY --failed-entries FAILED --output-dir DIR
    python3 retry_failed_descriptions.py auto --api-key KEY --input INPUT --output OUTPUT --output-dir DIR
    python3 retry_failed_descriptions.py merge --original ORIG --retry RETRY --merged MERGED
"""

import sys
import os
import json
import time
import shutil
import argparse
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import functions from req.py
try:
    from synthsis.req import (
        encode_image,
        make_api_request,
        extract_description,
        process_single_image,
        SYSTEM_INSTRUCTION,
        MAX_RETRIES,
        BASE_DELAY
    )
except ImportError as e:
    print(f"Error: Could not import from req.py: {e}")
    print("Ensure req.py is in the same directory or PYTHONPATH")
    sys.exit(1)


# --- CONFIGURATION ---
DEFAULT_WORKERS = 5  # Conservative for retries (half of req.py's 10)
DEFAULT_MAX_RETRIES = 5
DEFAULT_MERGE_STRATEGY = 'retry_priority'
DEFAULT_REQUEST_DELAY = 0.0  # Additional delay between requests (seconds)


def load_jsonl_as_dict(jsonl_path, key_field='path'):
    """
    Load JSONL file into dictionary indexed by key field.

    Args:
        jsonl_path: Path to JSONL file
        key_field: Field to use as dictionary key (default: 'path')

    Returns:
        dict: {key_value: entry_dict}
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    entries_dict = {}
    duplicates = []
    skipped_lines = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                key = entry.get(key_field)

                if key is None:
                    print(f"Warning: Line {line_num} missing '{key_field}' field, skipping")
                    skipped_lines += 1
                    continue

                if key in entries_dict:
                    duplicates.append(key)

                entries_dict[key] = entry

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                skipped_lines += 1

    if duplicates:
        print(f"Warning: Found {len(duplicates)} duplicate keys (keeping last occurrence)")

    if skipped_lines > 0:
        print(f"Loaded {len(entries_dict)} entries, skipped {skipped_lines} invalid/incomplete lines")

    return entries_dict


def identify_failed_entries(input_jsonl, output_jsonl, key_field='path'):
    """
    Identify entries that failed during processing (present in input but not output).

    Args:
        input_jsonl: Path to input JSONL file (all entries)
        output_jsonl: Path to output JSONL file (successful entries)
        key_field: Field to use for comparison (default: 'path')

    Returns:
        list: Failed entries
    """
    print(f"\nLoading input entries from: {input_jsonl}")
    input_dict = load_jsonl_as_dict(input_jsonl, key_field)

    print(f"Loading output entries from: {output_jsonl}")
    output_dict = load_jsonl_as_dict(output_jsonl, key_field)

    # Find entries in input but not in output
    failed_entries = []
    for key, entry in input_dict.items():
        if key not in output_dict:
            failed_entries.append(entry)

    # Print statistics
    total_input = len(input_dict)
    successful = len(output_dict)
    failed = len(failed_entries)

    print("\n" + "=" * 80)
    print("Identification Results")
    print("=" * 80)
    print(f"Total input entries: {total_input}")
    print(f"Successful entries: {successful} ({successful/total_input*100:.1f}%)")
    print(f"Failed entries: {failed} ({failed/total_input*100:.1f}%)")
    print("=" * 80)

    return failed_entries


def save_failed_entries(failed_entries, output_path):
    """
    Save failed entries to JSONL file.

    Args:
        failed_entries: List of failed entry dicts
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in failed_entries:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"\nSaved {len(failed_entries)} failed entries to: {output_path}")


def validate_entry(entry, check_description=False):
    """
    Validate entry has required fields.

    Args:
        entry: Entry dict to validate
        check_description: If True, also check for reasoning_description field

    Returns:
        tuple: (is_valid: bool, message: str)
    """
    required = ['path', 'id', 'ocr_text', 'detected_objects', 'bin']

    for field in required:
        if field not in entry:
            return False, f"Missing required field: {field}"

    # For output validation: check reasoning_description
    if check_description:
        if 'reasoning_description' not in entry:
            return False, "Missing reasoning_description field"

        if not entry['reasoning_description'].strip():
            return False, "Empty reasoning_description"

    return True, "Valid"


def retry_failed_entries(failed_jsonl, api_key, output_dir, workers=DEFAULT_WORKERS, max_retries=DEFAULT_MAX_RETRIES, request_delay=DEFAULT_REQUEST_DELAY):
    """
    Retry processing failed entries using req.py infrastructure.

    Args:
        failed_jsonl: Path to JSONL file with failed entries
        api_key: OpenRouter API key
        output_dir: Output directory for results
        workers: Number of parallel workers (default: 5)
        max_retries: Maximum API retry attempts per entry (default: 5)
        request_delay: Additional delay between requests in seconds (default: 0.0)

    Returns:
        tuple: (success_count, failure_count)
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = output_dir / "all_binned_images_with_descriptions.jsonl"
    error_log = output_dir / "retry_errors.log"

    # Load failed entries
    print(f"\nLoading failed entries from: {failed_jsonl}")

    if not os.path.exists(failed_jsonl):
        raise FileNotFoundError(f"Failed entries file not found: {failed_jsonl}")

    entries = []
    with open(failed_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not entries:
        print("No entries to retry. Exiting.")
        return 0, 0

    total = len(entries)
    print(f"\nRetrying {total} entries with {workers} workers...")
    print(f"Output will be saved to: {output_jsonl}")
    print(f"Errors will be logged to: {error_log}")
    print()

    # Process entries in parallel
    results = [None] * total
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_image,
                entry,
                api_key,
                SYSTEM_INSTRUCTION,
                idx
            ): idx
            for idx, entry in enumerate(entries)
        }

        # Collect results with progress bar
        with tqdm(total=total, desc="Retrying failed entries") as pbar:
            for future in as_completed(futures):
                result = future.result()
                idx = result["index"]
                results[idx] = result

                # Update progress bar description
                if result["status"] == "error":
                    pbar.set_postfix({"last_error": result["error"][:50]})

                pbar.update(1)

                # Optional delay between requests
                if request_delay > 0:
                    time.sleep(request_delay)

    elapsed_time = time.time() - start_time

    # Write outputs
    print("\nWriting results...")

    # Write successful entries
    success_count = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for result in results:
            if result["status"] == "success" and result["entry"] is not None:
                # Validate before writing
                is_valid, msg = validate_entry(result["entry"], check_description=True)
                if is_valid:
                    json_line = json.dumps(result["entry"], ensure_ascii=False)
                    f.write(json_line + "\n")
                    success_count += 1
                else:
                    print(f"Warning: Skipping invalid entry: {msg}")

    print(f"Wrote {success_count} enhanced entries to {output_jsonl}")

    # Write error log
    error_count = 0
    with open(error_log, 'w', encoding='utf-8') as f:
        f.write("Error Log - Retry Processing\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            if result["status"] == "error":
                error_count += 1
                f.write(f"Index: {result['index']}\n")
                f.write(f"Error: {result['error']}\n")
                f.write("-" * 80 + "\n")

    if error_count > 0:
        print(f"Logged {error_count} errors to {error_log}")

    # Generate statistics
    print("\n" + "=" * 80)
    print("Retry Processing Complete")
    print("=" * 80)
    print(f"Total entries: {total}")
    print(f"Successful: {success_count} ({success_count/total*100:.1f}%)")
    print(f"Failed: {error_count} ({error_count/total*100:.1f}%)")
    print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"Average time per entry: {elapsed_time/total:.2f}s")
    print("=" * 80)

    return success_count, error_count


def merge_results(original_output, retry_output, merged_output, key_field='path', merge_strategy=DEFAULT_MERGE_STRATEGY):
    """
    Merge original and retry results with conflict resolution.

    Args:
        original_output: Path to original output JSONL
        retry_output: Path to retry output JSONL
        merged_output: Path to write merged JSONL
        key_field: Field to use for deduplication (default: 'path')
        merge_strategy: Conflict resolution strategy:
            - 'retry_priority': Retry results override original (default)
            - 'original_priority': Original results override retry

    Returns:
        int: Total merged entry count
    """
    print(f"\nLoading original results from: {original_output}")
    original_dict = load_jsonl_as_dict(original_output, key_field)

    print(f"Loading retry results from: {retry_output}")
    retry_dict = load_jsonl_as_dict(retry_output, key_field)

    # Combine with conflict resolution
    merged = dict(original_dict)  # Start with all original entries
    conflicts = 0
    new_entries = 0

    for key, retry_entry in retry_dict.items():
        if key in merged:
            # Conflict: entry exists in both
            conflicts += 1
            if merge_strategy == 'original_priority':
                continue  # Keep original
            else:  # 'retry_priority'
                merged[key] = retry_entry  # Override with retry
        else:
            # New entry from retry
            new_entries += 1
            merged[key] = retry_entry

    # Create backup if output file already exists
    if os.path.exists(merged_output):
        backup_path = merged_output + f".backup_{int(time.time())}"
        shutil.copy2(merged_output, backup_path)
        print(f"Created backup: {backup_path}")

    # Write merged results using atomic write
    temp_fd, temp_path = tempfile.mkstemp(suffix='.jsonl', dir=os.path.dirname(merged_output) or '.')

    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            for entry in merged.values():
                # Validate before writing
                is_valid, msg = validate_entry(entry, check_description=True)
                if is_valid:
                    json_line = json.dumps(entry, ensure_ascii=False)
                    f.write(json_line + "\n")
                else:
                    print(f"Warning: Skipping invalid entry {entry.get('path')}: {msg}")

        # Atomic rename
        shutil.move(temp_path, merged_output)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"Failed to write merged output: {e}")

    # Print statistics
    print("\n" + "=" * 80)
    print("Merge Complete")
    print("=" * 80)
    print(f"Original entries: {len(original_dict)}")
    print(f"Retry entries: {len(retry_dict)}")
    print(f"  - New entries from retry: {new_entries}")
    print(f"  - Conflicts (resolved with '{merge_strategy}'): {conflicts}")
    print(f"Total merged entries: {len(merged)}")
    print(f"Wrote merged results to: {merged_output}")
    print("=" * 80)

    return len(merged)


def cmd_identify(args):
    """Handle 'identify' subcommand."""
    failed_entries = identify_failed_entries(args.input, args.output, args.key_field)

    if failed_entries:
        save_failed_entries(failed_entries, args.save_failed)
    else:
        print("\nNo failed entries found. All entries processed successfully!")


def cmd_retry(args):
    """Handle 'retry' subcommand."""
    success_count, error_count = retry_failed_entries(
        args.failed_entries,
        args.api_key,
        args.output_dir,
        args.workers,
        args.max_retries,
        args.request_delay
    )

    if error_count > 0:
        print(f"\n{error_count} entries still failed. You can:")
        print(f"1. Run 'identify' again on the retry output to find still-failed entries")
        print(f"2. Run 'retry' again with fewer workers or longer delays")


def cmd_auto(args):
    """Handle 'auto' subcommand (identify + retry + optional merge)."""
    print("=" * 80)
    print("AUTO MODE: Identify + Retry + Merge")
    print("=" * 80)

    # Step 1: Identify failed entries
    print("\n[Step 1/3] Identifying failed entries...")
    failed_entries = identify_failed_entries(args.input, args.output, args.key_field)

    if not failed_entries:
        print("\nNo failed entries found. All entries processed successfully!")
        return

    # Save failed entries to temp file
    temp_failed_path = Path(args.output_dir) / "temp_failed_entries.jsonl"
    save_failed_entries(failed_entries, temp_failed_path)

    # Step 2: Retry failed entries
    print("\n[Step 2/3] Retrying failed entries...")
    success_count, error_count = retry_failed_entries(
        temp_failed_path,
        args.api_key,
        args.output_dir,
        args.workers,
        args.max_retries,
        args.request_delay
    )

    if success_count == 0:
        print("\nNo entries were successfully retried. Skipping merge step.")
        return

    # Step 3: Merge results (if merge_output specified)
    if args.merge_output:
        print("\n[Step 3/3] Merging results...")
        retry_output = Path(args.output_dir) / "all_binned_images_with_descriptions.jsonl"
        merge_results(
            args.output,
            retry_output,
            args.merge_output,
            args.key_field,
            args.merge_strategy
        )
    else:
        print("\n[Step 3/3] Skipping merge (no --merge-output specified)")
        print(f"Retry results saved to: {Path(args.output_dir) / 'all_binned_images_with_descriptions.jsonl'}")

    # Cleanup temp file
    if temp_failed_path.exists():
        temp_failed_path.unlink()

    print("\nAuto mode complete!")


def cmd_merge(args):
    """Handle 'merge' subcommand."""
    merge_results(
        args.original,
        args.retry,
        args.merged,
        args.key_field,
        args.strategy
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Retry Failed Image Descriptions - Identify and retry failed API processing entries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Identify failed entries
  python3 retry_failed_descriptions.py identify \\
    --input all_binned_images.jsonl \\
    --output all_binned_images_with_descriptions.jsonl \\
    --save-failed failed_entries.jsonl

  # Retry failed entries
  python3 retry_failed_descriptions.py retry \\
    --api-key sk-or-v1-xxxxx \\
    --failed-entries failed_entries.jsonl \\
    --output-dir retry_results \\
    --workers 5 \\
    --request-delay 1.0  # Add 1 second delay between requests

  # Auto mode (identify + retry + merge)
  python3 retry_failed_descriptions.py auto \\
    --api-key sk-or-v1-xxxxx \\
    --input all_binned_images.jsonl \\
    --output all_binned_images_with_descriptions.jsonl \\
    --output-dir retry_results \\
    --merge-output merged_descriptions.jsonl

  # Merge results
  python3 retry_failed_descriptions.py merge \\
    --original all_binned_images_with_descriptions.jsonl \\
    --retry retry_results/all_binned_images_with_descriptions.jsonl \\
    --merged merged_descriptions.jsonl
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    # Identify subcommand
    parser_identify = subparsers.add_parser('identify', help='Identify failed entries')
    parser_identify.add_argument('--input', required=True, help='Path to input JSONL file (all entries)')
    parser_identify.add_argument('--output', required=True, help='Path to output JSONL file (successful entries)')
    parser_identify.add_argument('--save-failed', required=True, help='Path to save failed entries JSONL')
    parser_identify.add_argument('--key-field', default='path', help='Field to use for comparison (default: path)')
    parser_identify.set_defaults(func=cmd_identify)

    # Retry subcommand
    parser_retry = subparsers.add_parser('retry', help='Retry failed entries')
    parser_retry.add_argument('--api-key', required=True, help='OpenRouter API key')
    parser_retry.add_argument('--failed-entries', required=True, help='Path to failed entries JSONL file')
    parser_retry.add_argument('--output-dir', required=True, help='Output directory for retry results')
    parser_retry.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help=f'Number of parallel workers (default: {DEFAULT_WORKERS})')
    parser_retry.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES, help=f'Maximum API retry attempts (default: {DEFAULT_MAX_RETRIES})')
    parser_retry.add_argument('--request-delay', type=float, default=DEFAULT_REQUEST_DELAY, help=f'Additional delay between requests in seconds (default: {DEFAULT_REQUEST_DELAY})')
    parser_retry.set_defaults(func=cmd_retry)

    # Auto subcommand
    parser_auto = subparsers.add_parser('auto', help='Auto mode: identify + retry + merge')
    parser_auto.add_argument('--api-key', required=True, help='OpenRouter API key')
    parser_auto.add_argument('--input', required=True, help='Path to input JSONL file (all entries)')
    parser_auto.add_argument('--output', required=True, help='Path to output JSONL file (successful entries)')
    parser_auto.add_argument('--output-dir', required=True, help='Output directory for retry results')
    parser_auto.add_argument('--merge-output', help='Path to save merged results (optional)')
    parser_auto.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help=f'Number of parallel workers (default: {DEFAULT_WORKERS})')
    parser_auto.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES, help=f'Maximum API retry attempts (default: {DEFAULT_MAX_RETRIES})')
    parser_auto.add_argument('--request-delay', type=float, default=DEFAULT_REQUEST_DELAY, help=f'Additional delay between requests in seconds (default: {DEFAULT_REQUEST_DELAY})')
    parser_auto.add_argument('--key-field', default='path', help='Field to use for comparison (default: path)')
    parser_auto.add_argument('--merge-strategy', default=DEFAULT_MERGE_STRATEGY, choices=['retry_priority', 'original_priority'], help=f'Merge conflict resolution (default: {DEFAULT_MERGE_STRATEGY})')
    parser_auto.set_defaults(func=cmd_auto)

    # Merge subcommand
    parser_merge = subparsers.add_parser('merge', help='Merge original and retry results')
    parser_merge.add_argument('--original', required=True, help='Path to original output JSONL')
    parser_merge.add_argument('--retry', required=True, help='Path to retry output JSONL')
    parser_merge.add_argument('--merged', required=True, help='Path to save merged JSONL')
    parser_merge.add_argument('--key-field', default='path', help='Field to use for deduplication (default: path)')
    parser_merge.add_argument('--strategy', default=DEFAULT_MERGE_STRATEGY, choices=['retry_priority', 'original_priority'], help=f'Merge conflict resolution (default: {DEFAULT_MERGE_STRATEGY})')
    parser_merge.set_defaults(func=cmd_merge)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
