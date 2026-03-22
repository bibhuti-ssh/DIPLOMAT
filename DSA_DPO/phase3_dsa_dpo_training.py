"""
Phase 3: DSA-DPO Training Data Preparation & Training

This script:
1. Converts segment pairs from Phase 2 into LLaMA-Factory DSA-DPO training format
2. Updates dataset_info.json to register the new dataset
3. Provides instructions for running DSA-DPO training with the prepared data

The DSA-DPO format (based on LLaMA-Factory):
- "conversations": context before the diverging turn (list of user/assistant messages)
- "chosen": the positive employer response
- "rejected": the negative employer response
"""

import os
import json
import argparse
from typing import Dict, List
import shutil
from datetime import datetime


class DsaDpoTrainingPrep:
    """
    Prepares DSA-DPO training data from Phase 2 segment pairs.
    """
    
    def __init__(
        self,
        segment_pairs_path: str = "dsa_dpo_pipeline/outputs/phase2_output/dsa_dpo_segment_pairs.json",
        output_dir: str = "training/LLaMA-Factory/data",
        dataset_name: str = "hr_dsa_dpo_segments",
    ):
        self.segment_pairs_path = segment_pairs_path
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.output_file = os.path.join(output_dir, f"{dataset_name}.json")
        self.dataset_info_path = os.path.join(output_dir, "dataset_info.json")
        
        print("=" * 70)
        print("PHASE 3: DSA-DPO Training Data Preparation")
        print("=" * 70)
    
    def load_segment_pairs(self) -> List[Dict]:
        """Load segment pairs from Phase 2."""
        print(f"\nLoading segment pairs from: {self.segment_pairs_path}")
        
        if not os.path.exists(self.segment_pairs_path):
            raise FileNotFoundError(f"Segment pairs file not found: {self.segment_pairs_path}")
        
        with open(self.segment_pairs_path, "r", encoding="utf-8") as f:
            segment_pairs = json.load(f)
        
        print(f"✓ Loaded {len(segment_pairs)} segment pairs")
        return segment_pairs
    
    def convert_to_dsa_dpo_format(self, segment_pairs: List[Dict]) -> List[Dict]:
        """
        Convert segment pairs to LLaMA-Factory DSA-DPO format.
        
        Format:
        {
            "conversations": [  // Context before divergence
                {"from": "human", "value": "candidate utterance"},
                {"from": "gpt", "value": "employer utterance"},
                ...
            ],
            "chosen": {"from": "gpt", "value": "positive employer response"},
            "rejected": {"from": "gpt", "value": "negative employer response"}
        }
        """
        print("\nConverting to DSA-DPO training format...")
        
        dsa_dpo_data = []
        skipped = 0
        
        for sp in segment_pairs:
            try:
                # Extract data
                context_before = sp.get("context_before", [])
                positive_segment = sp.get("positive_segment", [])
                negative_segment = sp.get("negative_segment", [])
                
                if not positive_segment or not negative_segment:
                    skipped += 1
                    continue
                
                # Build conversation context (everything before the error turn)
                conversations = []
                for turn in context_before:
                    role = turn["role"]
                    content = turn["content"]
                    
                    # Map roles to LLaMA-Factory format
                    from_role = "human" if role == "candidate" else "gpt"
                    conversations.append({
                        "from": from_role,
                        "value": content
                    })
                
                # Find the first employer turn in segments
                first_employer_idx_pos = None
                for i, turn in enumerate(positive_segment):
                    if turn["role"] == "employer":
                        first_employer_idx_pos = i
                        break
                
                first_employer_idx_neg = None
                for i, turn in enumerate(negative_segment):
                    if turn["role"] == "employer":
                        first_employer_idx_neg = i
                        break
                
                if first_employer_idx_pos is None or first_employer_idx_neg is None:
                    print(f"  ⚠ Skipping pair {sp.get('pair_id', 'unknown')}: no employer turn in segment")
                    skipped += 1
                    continue
                
                # Add candidate turns before the first employer response to context
                for i in range(first_employer_idx_pos):
                    turn = positive_segment[i]
                    if turn["role"] == "candidate":
                        conversations.append({
                            "from": "human",
                            "value": turn["content"]
                        })
                
                # The first employer turn is the divergence point
                positive_response = positive_segment[first_employer_idx_pos]["content"]
                negative_response = negative_segment[first_employer_idx_neg]["content"]
                
                # If there are multiple employer turns, concatenate them
                # (This captures full employer strategy across multiple turns)
                if len(positive_segment) > first_employer_idx_pos + 1:
                    for i in range(first_employer_idx_pos + 1, len(positive_segment)):
                        turn = positive_segment[i]
                        if turn["role"] == "employer":
                            positive_response += "\n\n" + turn["content"]
                
                if len(negative_segment) > first_employer_idx_neg + 1:
                    for i in range(first_employer_idx_neg + 1, len(negative_segment)):
                        turn = negative_segment[i]
                        if turn["role"] == "employer":
                            negative_response += "\n\n" + turn["content"]
                
                # Create DSA-DPO training example
                dsa_dpo_example = {
                    "conversations": conversations,
                    "chosen": {
                        "from": "gpt",
                        "value": positive_response
                    },
                    "rejected": {
                        "from": "gpt",
                        "value": negative_response
                    },
                    # Metadata for reference (not used in training)
                    "_metadata": {
                        "pair_id": sp.get("pair_id", ""),
                        "segment_id": sp.get("segment_id", ""),
                        "scenario": sp.get("scenario", {}).get("domain", ""),
                        "selection_reason": sp.get("selection_reason", ""),
                        "score_improvement": sp.get("score_improvement", 0),
                    }
                }
                
                dsa_dpo_data.append(dsa_dpo_example)
                
            except Exception as e:
                print(f"  ⚠ Error processing pair {sp.get('pair_id', 'unknown')}: {e}")
                skipped += 1
                continue
        
        print(f"✓ Converted {len(dsa_dpo_data)} pairs")
        if skipped > 0:
            print(f"  ⚠ Skipped {skipped} pairs due to errors")
        
        return dsa_dpo_data
    
    def save_training_data(self, dsa_dpo_data: List[Dict]):
        """Save DSA-DPO training data to file."""
        print(f"\nSaving training data to: {self.output_file}")
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(dsa_dpo_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(dsa_dpo_data)} training examples")
    
    def update_dataset_info(self):
        """Update dataset_info.json to register the new dataset."""
        print(f"\nUpdating dataset_info.json...")
        
        if not os.path.exists(self.dataset_info_path):
            print(f"  ⚠ dataset_info.json not found at {self.dataset_info_path}")
            print(f"  Creating new dataset_info.json")
            dataset_info = {}
        else:
            with open(self.dataset_info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        
        # Add our dataset entry
        dataset_info[self.dataset_name] = {
            "file_name": f"{self.dataset_name}.json",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        }
        
        # Backup original
        if os.path.exists(self.dataset_info_path):
            backup_path = f"{self.dataset_info_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(self.dataset_info_path, backup_path)
            print(f"  ✓ Backed up original to: {backup_path}")
        
        # Save updated dataset_info
        with open(self.dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Updated dataset_info.json with '{self.dataset_name}' dataset")
    
    def create_training_config(
        self,
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        adapter_name_or_path: str = "./trained_models/bc_lora_adapter",
        template: str = "mistral",
        output_dir: str = "saves/mistral-7b/lora/hr_dsa_dpo",
    ):
        """Create a training configuration file for DSA-DPO."""
        config_dir = "training/LLaMA-Factory/examples/train_lora"
        config_path = os.path.join(config_dir, "hr_dsa_dpo_training.yaml")
        
        config_content = f"""### model
# Base model + BC LoRA adapter loaded as starting point
model_name_or_path: {base_model}
adapter_name_or_path: {adapter_name_or_path}

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all
# Create a NEW LoRA on top of the BC-initialized weights
create_new_adapter: true
pref_beta: 0.1
pref_loss: sigmoid

### dataset
dataset: {self.dataset_name}
template: {template}
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
"""
        
        # Save config
        os.makedirs(config_dir, exist_ok=True)
        with open(config_path, "w") as f:
            f.write(config_content)
        
        print(f"\n✓ Created training config: {config_path}")
        return config_path
    
    def print_instructions(self, config_path: str, num_examples: int):
        """Print instructions for running DSA-DPO training."""
        print("\n" + "=" * 70)
        print("DSA-DPO TRAINING READY")
        print("=" * 70)
        print(f"\n✅ Training data prepared: {num_examples} examples")
        print(f"✅ Dataset registered: {self.dataset_name}")
        print(f"✅ Training config created: {config_path}")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS: Run DSA-DPO Training")
        print("=" * 70)
        
        print("\n1. Navigate to LLaMA-Factory directory:")
        print(f"   cd training/LLaMA-Factory")
        
        print("\n2. Run training:")
        print(f"   llamafactory-cli train examples/train_lora/hr_dsa_dpo_training.yaml")
        
        print("\n   OR with torchrun for multi-GPU:")
        print(f"   FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/hr_dsa_dpo_training.yaml")
        
        print("\n3. Monitor training:")
        print(f"   - Logs: saves/mistral-7b/lora/hr_dsa_dpo/")
        print(f"   - TensorBoard: tensorboard --logdir saves/mistral-7b/lora/hr_dsa_dpo/")
        
        print("\n4. After training, merge LoRA weights:")
        print(f"   llamafactory-cli export \\")
        print(f"     --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \\")
        print(f"     --adapter_name_or_path saves/mistral-7b/lora/hr_dsa_dpo \\")
        print(f"     --export_dir models/hr_dsa_dpo_mistral_final \\")
        print(f"     --export_size 2 \\")
        print(f"     --export_legacy_format false")
        
        print("\n" + "=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print(f"- Base Model: mistralai/Mistral-7B-Instruct-v0.3")
        print(f"- BC Adapter: ./trained_models/bc_lora_adapter")
        print(f"- Method: DSA-DPO (DPO with sigmoid loss)")
        print(f"- LoRA: Applied to all linear layers")
        print(f"- Beta: 0.1 (DPO temperature)")
        print(f"- Epochs: 3")
        print(f"- Learning rate: 5e-6")
        print(f"- Batch size: 1 x 8 grad accum = effective 8")
        
        print("\n" + "=" * 70)
    
    def run(self):
        """Run the complete Phase 3 pipeline."""
        try:
            # Load segment pairs
            segment_pairs = self.load_segment_pairs()
            
            # Convert to DSA-DPO format
            dsa_dpo_data = self.convert_to_dsa_dpo_format(segment_pairs)
            
            if not dsa_dpo_data:
                print("\n❌ No training data generated. Check Phase 2 output.")
                return
            
            # Save training data
            self.save_training_data(dsa_dpo_data)
            
            # Update dataset registry
            self.update_dataset_info()
            
            # Create training config
            config_path = self.create_training_config(
                base_model="mistralai/Mistral-7B-Instruct-v0.3",
                adapter_name_or_path="./trained_models/bc_lora_adapter",
                template="mistral",
                output_dir="saves/mistral-7b/lora/hr_dsa_dpo",
            )
            
            # Print instructions
            self.print_instructions(config_path, len(dsa_dpo_data))
            
            print("\n✅ Phase 3 complete! Ready for DSA-DPO training.")
            
        except Exception as e:
            print(f"\n❌ Error in Phase 3: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Phase 3: DSA-DPO Training Data Preparation")
    parser.add_argument(
        "--segment-pairs",
        type=str,
        default="dsa_dpo_pipeline/outputs/phase2_output/dsa_dpo_segment_pairs.json",
        help="Path to segment pairs from Phase 2"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/LLaMA-Factory/data",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="hr_dsa_dpo_segments",
        help="Name for the dataset"
    )
    
    args = parser.parse_args()
    
    prep = DsaDpoTrainingPrep(
        segment_pairs_path=args.segment_pairs,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )
    
    prep.run()


if __name__ == "__main__":
    main()
