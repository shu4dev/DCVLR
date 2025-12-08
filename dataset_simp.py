from datasets import load_dataset
from huggingface_hub import login
from PIL import Image as PILImage

# 1. (optional if you didn't do CLI login)
login()  # will prompt for your HF token once

# 2. Load dataset
ds = load_dataset("shu4dev/DCVLR_10K_details", split="train")

LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

def add_explicit_final_answer(example):
    ans_letter = example.get("answer", "").strip()
    options = example.get("options", [])

    idx = LETTER_TO_INDEX.get(ans_letter, None)

    if idx is not None and 0 <= idx < len(options):
        final_option_text = options[idx]
        base_trace = (example.get("reasoning_trace", "") or "").strip()
        example["reasoning_trace"] = f"{base_trace}\n\nFinal answer: {final_option_text}"
    else:
        example["reasoning_trace"] = example.get("reasoning_trace", "")

    return example

ds = ds.map(add_explicit_final_answer)

# ------------------------------------------------------------------
# Resize images in "path" so that the longest side <= 840 px
# "path" is an Image feature, so examples already contain PIL images.
# ------------------------------------------------------------------

MAX_LONG_SIDE = 840  # biggest side (width or height) in pixels

def resize_image_and_update_path(example):
    img = example.get("path")
    if img is None:
        return example

    # Case 1: Already a PIL image (most likely)
    if isinstance(img, PILImage.Image):
        w, h = img.size
        long_side = max(w, h)

        # Don't upscale; only shrink if needed
        if long_side > MAX_LONG_SIDE:
            scale = MAX_LONG_SIDE / float(long_side)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))

            img = img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

        example["path"] = img
        return example

    # Case 2: If it's a dict with a "path" key (decoded=False scenario)
    if isinstance(img, dict) and "path" in img and img["path"]:
        try:
            with PILImage.open(img["path"]) as pil_img:
                w, h = pil_img.size
                long_side = max(w, h)

                if long_side > MAX_LONG_SIDE:
                    scale = MAX_LONG_SIDE / float(long_side)
                    new_w = int(round(w * scale))
                    new_h = int(round(h * scale))

                    pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

                # Store back as a PIL image so HF handles it as Image
                example["path"] = pil_img
        except Exception as e:
            print(f"Failed to process image at {img['path']}: {e}")

    return example

ds = ds.map(resize_image_and_update_path, num_proc=4)

# 3. Keep only needed columns, then rename to image/problem/solution
ds = ds.select_columns(["path", "question", "reasoning_trace"])

ds = ds.rename_columns({
    "path": "image",
    "question": "problem",
    "reasoning_trace": "solution",
})

# (Optional, just to enforce column order explicitly)
ds = ds.select_columns(["image", "problem", "solution"])

# 4. Push to hub (choose your own repo_id)
repo_id = "shu4dev/DCVLR_10K"
ds.push_to_hub(repo_id, private=False)
