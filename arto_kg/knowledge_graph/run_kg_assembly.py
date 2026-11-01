"""Assemble TTL from a specification, its detections and its validation result.

Stage 4 only records: no image is opened here. All pixel-derived values come
from Stage 3's measurements.

An artwork whose validation result says "discarded" (it failed the style or
the object-presence gate) gets no TTL unless --include_discarded is given. With
--val_dir, an artwork without a complete validation result (no file, no steps,
or marked incomplete) gets no TTL either and is counted as failed; a result
without a status does not hold an artwork back.
"""

import os
import json
import logging
import argparse
import sys
sys.path.append(os.getcwd())
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from arto_kg.knowledge_graph.arto_mapper import ARTOMapper
from arto_kg.validation.result_envelope import is_complete

logger = logging.getLogger("TTLGen")


def setup_logger(log_file=None):
    """Log to the console, and to log_file when one is given."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

DISCARDED_PREFIX = "Discarded (failed at "


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
    """Convert one specification to TTL.

    config holds 'input_dir', 'od_dir', 'val_dir', 'output_dir' and
    'include_discarded'.
    """
    mapper = ARTOMapper() # One mapper per file
    
    # Paths
    original_path = os.path.join(config["input_dir"], filename)
    
    # Load Data
    data_original = load_json(original_path)
    if not data_original:
        return f"Skipped (Missing Original): {filename}"

    # Every file of an artwork is named by its artwork_id.
    base_id = data_original.get("artwork_id") or os.path.splitext(filename)[0]
    output_path = os.path.join(config["output_dir"], f"{base_id}.ttl")

    # OD file candidates
    data_od = None
    if config["od_dir"]:
        od_candidates = [
            # Stage 3 writes _fused; _combined is also accepted, as in Stage 3.
            f"{base_id}_fused.json",
            f"{base_id}_combined.json",
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
            # Subdirectory layout: val_dir/{artwork_id}/{artwork_id}_full_validation.json
            os.path.join(config["val_dir"], base_id, f"{base_id}_full_validation.json"),
        ]

        for cand_path in val_candidates:
            if os.path.exists(cand_path):
                data_validation = load_json(cand_path)
                break

        # With validation results given, an artwork has to have a complete one
        # (see result_envelope.is_complete).
        if not is_complete(data_validation):
            return f"Skipped (No complete validation result): {filename}"

    # Stage 3 discarded this artwork at a gate: it does not enter the graph.
    if (isinstance(data_validation, dict) and data_validation.get("status") == "discarded"
            and not config.get("include_discarded")):
        return f"{DISCARDED_PREFIX}{data_validation.get('failed_at') or 'unknown'}): {filename}"

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
    parser.add_argument("--output_dir", required=True, help="Directory to save generated TTL files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--include_discarded", action="store_true",
                        help="Also write TTL for artworks Stage 3 discarded at the style or "
                             "object-presence gate (skipped by default).")
    parser.add_argument("--log_file", default=None,
                        help="Also write the log to this file (console only by default).")

    args = parser.parse_args()
    setup_logger(args.log_file)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Check input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(2)

    # Only artwork specifications; other JSON (e.g. batch_summary.json) is skipped.
    files = [f for f in os.listdir(args.input_dir)
             if f.endswith(".json") and (f.startswith("artwork_")
                                         or f.startswith("batch_artwork_"))]
    skipped = len(os.listdir(args.input_dir)) - len(files)
    total_files = len(files)
    logger.info(f"Found {total_files} files to process in {args.input_dir}"
                + (f" ({skipped} non-artwork entries ignored)" if skipped else ""))
    
    if args.od_dir:
        logger.info(f"OD Directory: {args.od_dir}")
    if args.val_dir:
        logger.info(f"Validation Directory: {args.val_dir}")
        
    # Prepare config for workers
    config = {
        "input_dir": args.input_dir,
        "od_dir": args.od_dir,
        "val_dir": args.val_dir,
        "output_dir": args.output_dir,
        "include_discarded": args.include_discarded,
    }

    # Process in parallel
    results = []
    # Use functools.partial to pass config to all calls
    process_func = partial(process_single_file, config=config)
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(process_func, files):
            if result.startswith(("Error", "Skipped")):
                logger.warning(result)
            results.append(result)

            if len(results) % 100 == 0:
                logger.info(f"Processed {len(results)}/{total_files}")

    # Discarded artworks, by the gate they failed.
    discarded = Counter(r[len(DISCARDED_PREFIX):].split(")")[0]
                        for r in results if r.startswith(DISCARDED_PREFIX))
    if discarded:
        logger.info(f"Not written, discarded by Stage 3: {sum(discarded.values())} "
                    f"({', '.join(f'{gate}: {n}' for gate, n in discarded.most_common())}). "
                    "Use --include_discarded to write them too.")

    # A file that could not be read or converted makes the run fail.
    failed = [r for r in results if r.startswith(("Error", "Skipped"))]
    written = sum(1 for r in results if r.startswith("Success"))
    logger.info(f"Processing complete: {written} written, {sum(discarded.values())} discarded, "
                f"{len(failed)} failed.")
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
