#!/usr/bin/env python3
"""
Script to remove bucket_id and strategies_used fields from PROWESS.json
"""

import json
import os

def clean_prowess_dataset(filepath):
    """
    Remove bucket_id and strategies_used fields from PROWESS dataset
    """
    try:
        # Load the JSON file
        print(f"Loading {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if data is a list
        if isinstance(data, list):
            print(f"Processing {len(data)} records...")
            
            # Remove fields from each record
            for i, record in enumerate(data):
                if 'bucket_id' in record:
                    del record['bucket_id']
                if 'strategies_used' in record:
                    del record['strategies_used']
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1} records...")
            
            print(f"Completed processing all {len(data)} records")
        else:
            print("Warning: Data is not a list. Checking if it's a dict with records...")
            if isinstance(data, dict):
                for key, records in data.items():
                    if isinstance(records, list):
                        print(f"Processing '{key}' with {len(records)} records...")
                        for record in records:
                            if 'bucket_id' in record:
                                del record['bucket_id']
                            if 'strategies_used' in record:
                                del record['strategies_used']
        
        # Save the cleaned data back to the same file
        print(f"Saving cleaned data to {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print("✅ Successfully removed 'bucket_id' and 'strategies_used' fields!")
        print(f"✅ File saved: {filepath}")
        
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in file: {filepath}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prowess_file = os.path.join(script_dir, 'PROWESS.json')
    
    clean_prowess_dataset(prowess_file)
