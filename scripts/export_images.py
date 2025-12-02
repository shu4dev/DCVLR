#!/usr/bin/env python
"""
Export images from a list of Hugging Face datasets.

Requires:
    pip install datasets pillow requests tqdm
"""

from pathlib import Path
import os
import re
import requests
from io import BytesIO
import zipfile

from datasets import load_dataset, DatasetDict, Image as HFImage
from PIL import Image
from tqdm.auto import tqdm


REPO_IDS = [
    "Luckyjhg/Geo170K",
    "Zhiqiang007/MathV360K",
]

ROOT_OUT = Path("./data")


def sanitize_repo_id(repo_id: str) -> str:
    # Turn "org/name" into "org__name"
    return repo_id.replace("/", "__")


def detect_image_column(ds) -> str:
    """
    Try to detect which column contains images or image URLs.
    Priority:
      1. HF Image feature
      2. Column name containing 'image' or 'img'
    """
    # 1) Check features for HF Image type
    for name, feature in ds.features.items():
        if isinstance(feature, HFImage):
            return name

    # 2) Heuristic on column names
    candidates = []
    for name in ds.column_names:
        lower = name.lower()
        if "image" in lower or lower == "img" or "img_path" in lower:
            candidates.append(name)

    if candidates:
        # choose the shortest name as a heuristic
        return sorted(candidates, key=len)[0]

    raise ValueError(f"Could not find an image column in columns: {ds.column_names}")


def is_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))


def download_image_from_url(url: str, timeout: int = 20) -> Image.Image:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def extract_image_from_zip(path_str: str, extract_dir: Path = None) -> Image.Image:
    """
    Extract an image from a zip file.
    Handles paths in formats like:
      - "zip://path/to/file.zip::path/inside.jpg"
      - "path/to/file.zip::path/inside.jpg"

    If extract_dir is provided, extracts the entire zip file to that directory
    and then loads the image from the extracted location.
    """
    # Parse the zip path
    if path_str.startswith("zip://"):
        path_str = path_str[6:]  # Remove "zip://" prefix

    if "::" in path_str:
        zip_path, inner_path = path_str.split("::", 1)
    else:
        # Try to detect .zip in the path
        parts = path_str.split(".zip")
        if len(parts) >= 2:
            zip_path = parts[0] + ".zip"
            inner_path = parts[1].lstrip("/\\")
        else:
            raise ValueError(f"Cannot parse zip path: {path_str}")

    # If extract_dir is provided, extract the zip file first
    if extract_dir:
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted_image_path = extract_dir / inner_path

        # Check if already extracted
        if not extracted_image_path.exists():
            # Extract the specific file or entire zip if not already done
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Check if we need to extract the whole zip
                marker_file = extract_dir / ".extracted"
                if not marker_file.exists():
                    print(f"  Extracting {zip_path} to {extract_dir}...")
                    zf.extractall(extract_dir)
                    marker_file.touch()
                elif inner_path not in zf.namelist():
                    raise FileNotFoundError(f"Image {inner_path} not found in {zip_path}")
                else:
                    # Extract just this file
                    zf.extract(inner_path, extract_dir)

        # Load from extracted location
        img = Image.open(extracted_image_path).convert("RGB")
        return img
    else:
        # Extract and open the image directly from zip (original behavior)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(inner_path) as img_file:
                img = Image.open(img_file).convert("RGB")
                # Load the image data immediately since the file will be closed
                img.load()
                return img


def save_image_value(value, out_path: Path, extract_dir: Path = None):
    """
    Handle different types of image-like values:
      - PIL.Image.Image
      - dict with 'path' or 'url'
      - string path or URL

    If extract_dir is provided, zip files will be extracted there.
    """
    img = None

    # Case 1: already a PIL image
    if isinstance(value, Image.Image):
        img = value

    # Case 2: dict with metadata
    elif isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            img = Image.open(BytesIO(value["bytes"])).convert("RGB")
        elif "url" in value and is_url(value["url"]):
            img = download_image_from_url(value["url"])
        elif "path" in value:
            path_val = value["path"]
            # Try to load from extracted directory first if it exists
            if extract_dir and extract_dir.exists():
                # Try to find the image in extracted directory by searching for the filename
                path_obj = Path(path_val)
                possible_paths = list(extract_dir.rglob(path_obj.name))
                if not possible_paths:
                    # Try with the full relative path structure (e.g., DVQA/images/file.png)
                    possible_paths = list(extract_dir.rglob(str(path_obj)))
                if possible_paths:
                    img = Image.open(possible_paths[0]).convert("RGB")

            # If not found in extracted dir, try original methods
            if img is None:
                if path_val.startswith("zip://") or "::" in path_val or ".zip" in path_val:
                    img = extract_image_from_zip(path_val, extract_dir)
                else:
                    img = Image.open(path_val).convert("RGB")

    # Case 3: plain string
    elif isinstance(value, str):
        if is_url(value):
            img = download_image_from_url(value)
        else:
            # Try to load from extracted directory first if it exists
            if extract_dir and extract_dir.exists():
                # Try to find the image in extracted directory by searching for the filename
                path_obj = Path(value)
                possible_paths = list(extract_dir.rglob(path_obj.name))
                if not possible_paths:
                    # Try with the full relative path structure
                    possible_paths = list(extract_dir.rglob(str(path_obj)))
                if possible_paths:
                    img = Image.open(possible_paths[0]).convert("RGB")

            # If not found in extracted dir, try original methods
            if img is None:
                # Check if it's a path inside a zip file
                if value.startswith("zip://") or "::" in value:
                    img = extract_image_from_zip(value, extract_dir)
                else:
                    img = Image.open(value).convert("RGB")

    if img is None:
        # Skip if we couldn't decode
        return False

    # Convert to RGB if needed (e.g., RGBA images can't be saved as JPEG)
    if img.mode != "RGB":
        img = img.convert("RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


def find_and_extract_zips(dataset_cache_dir: Path, extract_dir: Path):
    """
    Find all .zip files in the dataset cache directory and extract them.
    Returns True if any zips were found and extracted.
    """
    if not dataset_cache_dir.exists():
        return False

    zip_files = list(dataset_cache_dir.rglob("*.zip"))
    if not zip_files:
        return False

    extract_dir.mkdir(parents=True, exist_ok=True)
    marker_file = extract_dir / ".extracted"

    if marker_file.exists():
        return True  # Already extracted

    for zip_path in zip_files:
        print(f"  Found zip file: {zip_path.name}")
        print(f"  Extracting to {extract_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            print(f"  ! Failed to extract {zip_path}: {e}")

    marker_file.touch()
    return True


def export_split(repo_id: str, split_name: str, ds, out_dir: Path):
    img_col = detect_image_column(ds)
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Create an extract directory for zip files
    extract_dir = out_dir / "_extracted"

    # Try to find and extract zip files from the dataset cache
    try:
        # Get the cache directory from the dataset info
        if hasattr(ds, 'cache_files') and ds.cache_files:
            cache_file = Path(ds.cache_files[0]['filename'])
            dataset_cache_dir = cache_file.parent
            find_and_extract_zips(dataset_cache_dir, extract_dir)
    except Exception as e:
        print(f"  Note: Could not auto-extract zips: {e}")

    print(f"[{repo_id}] Exporting split '{split_name}' using column '{img_col}'")

    for idx, example in enumerate(tqdm(ds, desc=f"{repo_id} [{split_name}]")):
        value = example[img_col]
        out_path = split_dir / f"{idx:06d}.jpg"
        # skip existing to make re-runs cheaper
        if out_path.exists():
            continue
        try:
            ok = save_image_value(value, out_path, extract_dir)
        except Exception as e:
            print(f"  ! Error on index {idx}: {e}")
            ok = False

        if not ok:
            # you can log or debug here if needed
            pass


def export_repo(repo_id: str, root_out: Path):
    dataset_name = sanitize_repo_id(repo_id)
    out_dir = root_out / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing dataset: {repo_id} -> {out_dir} ===")

    # Load all splits (e.g. train/validation/test) if available
    ds_all = load_dataset(repo_id)

    if isinstance(ds_all, DatasetDict):
        for split_name, split_ds in ds_all.items():
            export_split(repo_id, split_name, split_ds, out_dir)
    else:
        # single split (treat as 'train')
        export_split(repo_id, "train", ds_all, out_dir)


def main():
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    for repo_id in REPO_IDS:
        try:
            export_repo(repo_id, ROOT_OUT)
        except Exception as e:
            print(f"*** Failed on {repo_id}: {e}")


if __name__ == "__main__":
    main()
