import os
import json
import logging
import argparse
import re
import sys
sys.path.append(os.getcwd())
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from arto_kg.knowledge_graph.arto_mapper import ARTOMapper

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
            
    # Validation file candidates
    data_validation = None
    if config["val_dir"]:
        val_candidates = [
            f"{base_id}_full_validation.json",
            f"{base_name}_full_validation.json"
        ]
        
        for cand in val_candidates:
            cand_path = os.path.join(config["val_dir"], cand)
            if os.path.exists(cand_path):
                data_validation = load_json(cand_path)
                break
            
    # if data_validation:
    #     print(f"DEBUG: Loaded validation data for {filename}")
    
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
        
    # Prepare config for workers
    config = {
        "input_dir": args.input_dir,
        "od_dir": args.od_dir,
        "val_dir": args.val_dir,
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
