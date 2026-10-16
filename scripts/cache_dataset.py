"""Pre-caches BCMID images resized to 224x224 RGB using multi-process parallelization.

Saves cached images to:
data/cached_224/mammogram/<filename>
data/cached_224/ultrasound/<filename>
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import BCMIDModalityDataset


def process_sample_task(args):
    image_path, filename, target_dir, size = args
    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path

    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB").resize(size, Image.Resampling.BILINEAR)
            rgb_img.save(target_path, "PNG", compress_level=1)
        return target_path
    except Exception as e:
        print(f"Error processing {filename}: {e}", flush=True)
        return None


def cache_modality(modality: str, max_workers: int = None):
    target_dir = os.path.join("data", "cached_224", modality)
    os.makedirs(target_dir, exist_ok=True)

    all_samples = []
    seen = set()
    for split in ["train", "val", "test"]:
        ds = BCMIDModalityDataset(split=split, modality=modality)
        for s in ds.samples:
            if s["filename"] not in seen:
                seen.add(s["filename"])
                all_samples.append((s["image_path"], s["filename"], target_dir, (224, 224)))

    workers = max_workers or os.cpu_count() or 8
    print(f"\nPre-caching {len(all_samples)} {modality} images using {workers} processes...", flush=True)
    count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for res in executor.map(process_sample_task, all_samples, chunksize=16):
            if res:
                count += 1
            if count % 200 == 0 or count == len(all_samples):
                print(f"  Processed {count}/{len(all_samples)} {modality} images...", flush=True)

    print(f"Successfully cached {count} {modality} images to {target_dir}!", flush=True)


if __name__ == "__main__":
    cache_modality("mammogram")
    cache_modality("ultrasound")
    print("\nDataset caching complete!", flush=True)
