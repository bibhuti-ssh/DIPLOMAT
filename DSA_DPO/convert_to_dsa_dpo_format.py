#!/usr/bin/env python3
"""
Convert DPO pairs (chosen/rejected) to DSA-DPO alternating format.

DSA-DPO expects alternating pairs: [rejected_0, chosen_0, rejected_1, chosen_1, ...]
Each entry is a complete conversation in SFT format (just conversations).

LLaMA-Factory expects conversations to start with "human" role, so we add a
system-like preamble if needed.
"""

import json
import argparse
from pathlib import Path


def convert_to_dsa_dpo_format(input_path: str, output_path: str):
    """Convert DPO pairs to DSA-DPO alternating format."""
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    dsa_dpo_data = []
    
    for item in data:
        context = item.get("conversations", [])
        chosen = item.get("chosen", {})
        rejected = item.get("rejected", {})
        
        # Skip if missing data
        if not context or not chosen or not rejected:
            print(f"Skipping item: missing data")
            continue
        
        # LLaMA-Factory expects conversations to start with "human"
        # If conversation starts with "gpt", add a human preamble
        if context and context[0].get("from") == "gpt":
            preamble = {
                "from": "human",
                "value": "Let's continue our HR negotiation conversation."
            }
            context = [preamble] + context
        
        # Create rejected conversation (context + rejected response)
        rejected_conv = {
            "conversations": context + [rejected]
        }
        
        # Create chosen conversation (context + chosen response)
        chosen_conv = {
            "conversations": context + [chosen]
        }
        
        # DSA-DPO expects alternating: rejected first, then chosen
        dsa_dpo_data.append(rejected_conv)
        dsa_dpo_data.append(chosen_conv)
    
    # Save converted data
    with open(output_path, 'w') as f:
        json.dump(dsa_dpo_data, f, indent=2)
    
    print(f"Converted {len(data)} pairs to {len(dsa_dpo_data)} entries")
    print(f"Saved to: {output_path}")
    
    return len(dsa_dpo_data)


def main():
    parser = argparse.ArgumentParser(description="Convert DPO pairs to DSA-DPO format")
    parser.add_argument("--input", "-i", 
                       default="./training/LLaMA-Factory/data/hr_dsa_dpo_segments.json",
                       help="Input file (DPO pairs format)")
    parser.add_argument("--output", "-o",
                       default="./training/LLaMA-Factory/data/hr_dsa_dpo_sft.json",
                       help="Output file (DSA-DPO alternating format)")
    
    args = parser.parse_args()
    
    convert_to_dsa_dpo_format(args.input, args.output)


if __name__ == "__main__":
    main()
