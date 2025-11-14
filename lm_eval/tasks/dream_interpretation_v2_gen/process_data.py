#!/usr/bin/env python3
"""
Script to process dream interpretation data files and filter for test split only.
"""

import json
import os

def filter_test_split(input_file, output_file):
    """
    Read a JSON file and filter to include only samples with split='test'.
    Preserves all metadata in the samples.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter for test split only
        test_data = [sample for sample in data if sample.get('split') == 'test']
        
        print(f"Processing {input_file}:")
        print(f"  Original samples: {len(data)}")
        print(f"  Test samples: {len(test_data)}")
        
        # Write filtered data to output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved to: {output_file}\n")
        return len(test_data)
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return 0
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {input_file}.")
        return 0
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return 0

def main():
    # Define source and destination paths
    source_dir = "/home/abdelrahman.sadallah/mbzuai/Jais-dream-interpretation/data/9-mcqs"
    dest_dir = "/home/abdelrahman.sadallah/mbzuai/lm-evaluation-harness/lm_eval/tasks/dream_interpretation_v2/data"
    
    # File mappings: source_filename -> destination_filename
    file_mappings = {
        "english.json": "english.json",
        "english-arabic.json": "english-arabic.json", 
        "arabic.json": "arabic.json",
        "arabic-english.json": "arabic-english.json"
    }
    
    total_processed = 0
    
    print("Processing dream interpretation data files...")
    print("=" * 60)
    
    for source_name, dest_name in file_mappings.items():
        source_path = os.path.join(source_dir, source_name)
        dest_path = os.path.join(dest_dir, dest_name)
        
        count = filter_test_split(source_path, dest_path)
        total_processed += count
    
    print("=" * 60)
    print(f"Processing complete! Total test samples processed: {total_processed}")

if __name__ == "__main__":
    main()