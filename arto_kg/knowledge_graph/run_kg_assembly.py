import os
import json
import logging
import argparse
import re
import sys
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from arto_kg.knowledge_graph.arto_mapper import ARTOMapper


def _load_color_vocab():
    """Load color_vocabulary.csv → [(r,g,b,name), ...]. Uses stdlib csv only."""
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "validation", "validators", "color_vocabulary.csv"
    )
    vocab = []
    try:
        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                hex_val = row.get('hex_string', '').strip()
                name = row.get('color_name', '').strip()
                if name and len(hex_val) == 6:
                    try:
                        r = int(hex_val[0:2], 16)
                        g = int(hex_val[2:4], 16)
                        b = int(hex_val[4:6], 16)
                        vocab.append((r, g, b, name))
                    except ValueError:
                        pass
    except Exception:
        pass  # Falls back to unnamed colors gracefully
    return vocab

_COLOR_VOCAB = _load_color_vocab()


def _rgb_to_color_name(rgb):
    """Return nearest color name from color_vocabulary.csv for [R, G, B]."""
    if not _COLOR_VOCAB:
        return "Color"
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    best_name, best_dist = "Color", float("inf")
    for cr, cg, cb, name in _COLOR_VOCAB:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


def extract_dominant_colors(image_path, bbox, n_colors=3):
    """Extract n dominant colors from the bbox region [x1,y1,x2,y2] of an image.
    Uses PIL quantize (no sklearn needed). Returns [(rgb_list, proportion), ...] or []."""
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return []

        crop = img.crop((x1, y1, x2, y2)).resize((50, 50))
        quantized = crop.quantize(colors=n_colors)
        palette = quantized.getpalette()  # flat [R,G,B, R,G,B, ...]

        pixels = np.array(quantized.getdata(), dtype=np.int32)
        counts = np.bincount(pixels, minlength=n_colors)[:n_colors]
        total = int(pixels.size)

        results = []
        for i in range(n_colors):
            rgb = [palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]]
            prop = round(float(counts[i]) / total, 4)
            results.append((rgb, prop))
        results.sort(key=lambda x: -x[1])
        return results
    except Exception as e:
        logger.warning(f"Color extraction failed for {image_path}: {e}")
        return []

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("ttl_generation.log")
        ]
    )
    return logging.getLogger("TTLGen")

logger = setup_logger()

def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        return None

def process_single_file(filename, config):
    """
    Process a single file with provided configuration paths.
    config: dict containing 'input_dir', 'od_dir', 'val_dir', 'output_dir'
    """
    mapper = ARTOMapper() # Instantiate per process
    
    # Paths
    original_path = os.path.join(config["input_dir"], filename)
    
    # Output path - change extension to .ttl
    ttl_filename = os.path.splitext(filename)[0] + ".ttl"
    output_path = os.path.join(config["output_dir"], ttl_filename)
    
    # Load Data
    data_original = load_json(original_path)
    if not data_original:
        return f"Skipped (Missing Original): {filename}"
        
    # Robust filename matching
    # filename e.g. artwork_20260103_020452_359_v2.json
    base_name = os.path.splitext(filename)[0]
    
    # Extract ID like artwork_20260103_020452_359 (without suffixes)
    match = re.search(r"(artwork_\d{8}_\d{6}_\d{3})", filename)
    base_id = match.group(1) if match else base_name
    
    # OD file candidates
    data_od = None
    if config["od_dir"]:
        od_candidates = [
            f"{base_id}_combined.json",
            f"{base_id}.json",
            filename 
        ]
        
        for cand in od_candidates:
            cand_path = os.path.join(config["od_dir"], cand)
            if os.path.exists(cand_path):
                data_od = load_json(cand_path)
                break
            
    # Validation file candidates - search both flat directory and artwork-ID subdirectory
    data_validation = None
    if config["val_dir"]:
        val_candidates = [
            os.path.join(config["val_dir"], f"{base_id}_full_validation.json"),
            os.path.join(config["val_dir"], f"{base_name}_full_validation.json"),
            # Subdirectory layout: val_dir/{artwork_id}/{artwork_id}_full_validation.json
            os.path.join(config["val_dir"], base_id, f"{base_id}_full_validation.json"),
            os.path.join(config["val_dir"], base_name, f"{base_name}_full_validation.json"),
        ]

        for cand_path in val_candidates:
            if os.path.exists(cand_path):
                data_validation = load_json(cand_path)
                break
            
    # Extract image-based dominant colors for each detected object bbox
    if config.get("image_dir") and data_od:
        image_path = os.path.join(config["image_dir"], f"{base_id}.png")
        if os.path.exists(image_path):
            image_colors = {}
            for obj in data_od.get("detected_objects", []):
                name = (obj.get("mapped_gt") or obj.get("label", "")).lower()
                bbox = obj.get("box")
                if name and bbox and name not in image_colors:
                    colors = extract_dominant_colors(image_path, bbox)
                    if colors:
                        # Attach color name from vocabulary: (rgb, proportion, name)
                        image_colors[name] = [
                            (rgb, prop, _rgb_to_color_name(rgb))
                            for rgb, prop in colors
                        ]
            if image_colors:
                data_od = dict(data_od)
                data_od["image_colors"] = image_colors

    try:
        mapper.convert_artwork_to_ttl(
            json_data=data_original,
            od_data=data_od,
            validation_data=data_validation,
            output_path=output_path
        )
        return f"Success: {filename}"
    except Exception as e:
        return f"Error processing {filename}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Generate ARTO Knowledge Graph (TTL) from JSON data.")
    
    parser.add_argument("--input_dir", required=True, help="Directory containing original generated JSON files.")
    parser.add_argument("--od_dir", default=None, help="Directory containing Object Detection results (optional).")
    parser.add_argument("--val_dir", default=None, help="Directory containing Validation results (optional).")
    parser.add_argument("--image_dir", default=None, help="Directory containing artwork images for RGB extraction (optional).")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated TTL files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return

    # Get list of files
    files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    total_files = len(files)
    logger.info(f"Found {total_files} files to process in {args.input_dir}")
    
    # Run matching stats (optional)
    if args.od_dir:
        logger.info(f"OD Directory: {args.od_dir}")
    if args.val_dir:
        logger.info(f"Validation Directory: {args.val_dir}")
    if args.image_dir:
        logger.info(f"Image Directory: {args.image_dir}")
        
    # Prepare config for workers
    config = {
        "input_dir": args.input_dir,
        "od_dir": args.od_dir,
        "val_dir": args.val_dir,
        "image_dir": args.image_dir,
        "output_dir": args.output_dir
    }

    # Process in parallel
    results = []
    # Use functools.partial to pass config to all calls
    process_func = partial(process_single_file, config=config)
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(process_func, files):
            if "Error" in result or "Skipped" in result:
                logger.warning(result)
            results.append(result)
            
            if len(results) % 100 == 0:
                logger.info(f"Processed {len(results)}/{total_files}")

    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
