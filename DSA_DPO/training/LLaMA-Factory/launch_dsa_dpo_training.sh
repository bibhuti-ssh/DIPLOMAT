#!/bin/bash
# DSA-DPO Training launch script (Mistral 7B)
# Ensure your conda environment is activated and LLaMA-Factory is installed

export WANDB_DISABLED=true

echo "=== Package versions in use ==="
python -c "import transformers, trl, accelerate, datasets; print('transformers', transformers.__version__); print('trl', trl.__version__); print('accelerate', accelerate.__version__); print('datasets', datasets.__version__)"
echo "==============================="

echo "Starting DSA-DPO training with hr_dsa_dpo_sft_pairs dataset..."
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/hr_dsa_dpo_training.yaml
