
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KEYS_TO_REMOVE = {
    "raw_output",
    "parsing_details",
    "processing_status",
    "metadata",
    "image_file",
    "json_file",
    "reconstruction_status",
    "primary_focus",
    "debug_info_v3"
}

def remove_keys_recursive(data, keys_to_remove):
    """
    Recursively removes keys from a dictionary or list of dictionaries.
    Returns True if any modification was made, False otherwise.
    """
    modified = False
    
    if isinstance(data, dict):
        # Create a list of keys to avoid runtime error during iteration
        for key in list(data.keys()):
            if key in keys_to_remove:
                del data[key]
                modified = True
            else:
                # Recurse into the value
                if remove_keys_recursive(data[key], keys_to_remove):
                    modified = True
                    
    elif isinstance(data, list):
        for item in data:
            if remove_keys_recursive(item, keys_to_remove):
                modified = True
                
    return modified

def clean_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if remove_keys_recursive(data, KEYS_TO_REMOVE):
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Cleaned: {file_path}")
        # else:
            # logger.info(f"Skipped (no changes): {file_path}")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Scanning directory: {base_dir}")
    
    # We will walk through the directory tree
    for root, dirs, files in os.walk(base_dir):
        # Exclude hidden directories/files just in case
        dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('_')]
        
        for file in files:
            if file.endswith(".json") and not file.endswith("_groundtruth.json"): 
                file_path = os.path.join(root, file)
                clean_json_file(file_path)

if __name__ == "__main__":
    main()
