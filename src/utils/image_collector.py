import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from tqdm import tqdm  # pip install tqdm


def _copy_one(index, entry, images_root, group_by_bin, move_instead_of_copy):
    """
    Copy a single image and return a result dict:
    {
        "index": index,
        "new_rel_path": "images/..." or "images/bin/..." (relative to per_json_dir),
        "error": "..." or None
    }
    """
    img_path_str = entry.get("path")
    img_id = entry.get("id")

    if not img_path_str:
        return {"index": index, "new_rel_path": None, "error": "no 'path' field"}

    src_path = Path(img_path_str)

    if not src_path.is_file():
        return {"index": index, "new_rel_path": None, "error": f"Image not found: {src_path}"}

    # Decide destination directory (under images_root)
    if group_by_bin:
        bin_label = entry.get("bin", "unknown")
        dest_dir = images_root / str(bin_label)
        rel_dir = Path("images") / str(bin_label)
    else:
        dest_dir = images_root
        rel_dir = Path("images")

    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = src_path.suffix
    dest_filename = f"{img_id}{suffix}" if img_id else src_path.name
    dest_path = dest_dir / dest_filename
    rel_path = rel_dir / dest_filename  # path to store in JSONL

    if dest_path.exists():
        # Treat this as success; file already there
        return {"index": index, "new_rel_path": str(rel_path), "error": None}

    try:
        if move_instead_of_copy:
            shutil.move(str(src_path), str(dest_path))
        else:
            shutil.copy2(str(src_path), str(dest_path))
    except Exception as e:
        return {"index": index, "new_rel_path": None, "error": str(e)}

    return {"index": index, "new_rel_path": str(rel_path), "error": None}


def collect_images_from_jsonl_fast(
    jsonl_file,
    output_root="output",
    group_by_bin=False,
    move_instead_of_copy=False,
    max_workers=8,
    keep_extension_folder=True,
):
    jsonl_path = Path(jsonl_file)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    # Folder: output/<json_filename>/ or output/<json_stem>/
    if keep_extension_folder:
        per_json_dir = Path(output_root) / jsonl_path.name
    else:
        per_json_dir = Path(output_root) / jsonl_path.stem

    images_root = per_json_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    # Load all entries first
    entries = []
    all_paths = []  # Track all image paths for duplicate detection
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Line {line_num}] Skipping invalid JSON: {e}")
                continue
            entries.append(entry)
            # Track the path for duplicate detection
            img_path = entry.get("path")
            if img_path:
                all_paths.append(img_path)

    total = len(entries)
    if total == 0:
        print("No valid entries found in JSONL.")
        return

    # Detect duplicate paths (appearing more than 2 times)
    path_counts = Counter(all_paths)
    duplicates = {path: count for path, count in path_counts.items() if count > 2}

    # Parallel I/O with progress bar
    results = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _copy_one,
                idx,
                entry,
                images_root,
                group_by_bin,
                move_instead_of_copy,
            )
            for idx, entry in enumerate(entries)
        ]

        with tqdm(total=total, desc=f"Processing {jsonl_path.name}") as pbar:
            for future in as_completed(futures):
                res = future.result()
                idx = res["index"]
                results[idx] = res
                if res["error"]:
                    # You can comment this out if you don't want error spam
                    print(f"[index {idx}] {res['error']}")
                pbar.update(1)

    # Write modified JSONL with updated paths
    output_jsonl_path = per_json_dir / jsonl_path.name
    num_success = 0
    num_total = 0
    missing_images = []  # Track all missing image paths

    with output_jsonl_path.open("w", encoding="utf-8") as out_f:
        for idx, (entry, res) in enumerate(zip(entries, results)):
            num_total += 1
            if res is None or res["new_rel_path"] is None:
                # Skip entries where we failed to copy
                if res and res["error"] and "Image not found" in res["error"]:
                    # Extract the path from the error message or use the original path
                    img_path = entry.get("path", "Unknown path")
                    missing_images.append(img_path)
                continue
            entry = dict(entry)  # shallow copy
            entry["path"] = res["new_rel_path"]  # e.g. "images/xxx.jpg" or "images/A/xxx.jpg"
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            num_success += 1

    print(f"\nFinished. Wrote {num_success}/{num_total} entries to {output_jsonl_path}")
    print(f"Images are under: {images_root}")

    # Print duplicate paths report
    if duplicates:
        print(f"\n⚠️  WARNING: {len(duplicates)} image path(s) appear more than 2 times:")
        for dup_path, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {dup_path} (appears {count} times)")
    else:
        print("\n✓ No duplicate paths found (all paths appear 2 or fewer times)")

    # Print summary of missing images
    if missing_images:
        print(f"\n⚠️  WARNING: {len(missing_images)} image(s) not found:")
        for img_path in missing_images:
            print(f"  - {img_path}")
    else:
        print("\n✓ All images found successfully!")


if __name__ == "__main__":
    collect_images_from_jsonl_fast(
        jsonl_file="./output/intermediate/stage4_qa_generation/hard_qa_pairs.jsonl",      # your input jsonl
        output_root="dataset",         # top-level output directory
        group_by_bin=False,           # True -> images/<bin>/...
        move_instead_of_copy=False,
        max_workers=96,
        keep_extension_folder=True,   # False -> folder is output/data instead of output/data.jsonl
    )