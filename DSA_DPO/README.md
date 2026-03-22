# DIPLOMAT & DSA-DPO Pipeline

Codebase for the **DIPLOMAT** model and the **DSA-DPO (Dialogue-State Adaptive Direct Preference Optimization)** pipeline. 

This repository allows you to automatically generate, score, and improve preference data (DPO pairs) for negotiation dialogues using LLM self-play. 

## Repository Structure

- `run_full_pipeline.sh`: **Main orchestrator.** Runs all 3 pipeline phases sequentially.
- `phase1_negative_generation.py`: Phase 1 script (Generates and evaluates baseline sessions).
- `phase2_positive_sampling.py`: Phase 2 script (Samples improved responses for bad sessions).
- `phase3_dsa_dpo_training.py`: Phase 3 script (Formats data for LLaMA-Factory).
- `session_generator.py`: Connects to LLMs (Gemini, OpenAI, Mistral) for dialogue generation.
- `session_scorer.py`: Calculates objective outcome scores (Satisfaction, Agreement, Conflict).
- `calculate_dsa_dpo_cost.py`: Estimates API costs/tokens prior to execution.
- `dsa_dpo_pipeline/outputs/`: Directory for all generated sessions, pairs, and metrics.
- `training/LLaMA-Factory/`: Embedded LLaMA-Factory configurations for DPO training.

## Setup & Execution

### 1. Requirements
Python 3.10+ required. 

**For Phase 1 & 2 (Data Generation & Evaluation):**
```bash
conda create --name diplomat --file requirements.txt
conda activate diplomat
```

**For Phase 3 (DPO Training via LLaMA-Factory):**
```bash
conda create --name diplomat_phase3 --file requirements_phase3.txt
conda activate diplomat_phase3
```

### 🐳 Docker Usage (Optional)
If you prefer not to install Conda locally, a comprehensive `Dockerfile` is provided that builds both environments natively on Linux.
```bash
docker build -t diplomat-pipeline .
docker run -it --gpus all diplomat-pipeline
```
Inside the container, simply use `conda activate diplomat` or `conda activate diplomat_phase3` as normal.

Set your API variables (required for the agents and judge):
```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
```

### 2. Configure Models
Edit `dsa_dpo_pipeline/config.yaml` to set your target LLMs for the candidate, employer, and judge roles.

### 3. Run Pipeline (Automated Mode)
Run the shell wrapper to execute all 3 phases sequentially.
```bash
# Set TARGET_SESSIONS inside the script to control output volume
bash run_full_pipeline.sh
```

### 4. Run Pipeline (Manual Mode)
You can execute phases manually for debugging or targeted generation.

**Phase 1: Negative Session Generation**
```bash
python phase1_negative_generation.py --config dsa_dpo_pipeline/config.yaml --target 300
```

**Phase 2: Positive Counterpart Sampling**
```bash
python phase2_positive_sampling.py --input dsa_dpo_pipeline/outputs/sessions/all_sessions.json --num-candidates 3
```

**Phase 3: DPO Dataset Prep (LLaMA-Factory Format)**
```bash
python phase3_dsa_dpo_training.py
```

## Training DIPLOMAT

Once Phase 3 finishes, your chosen/rejected data is compiled inside `training/LLaMA-Factory/data/hr_dsa_dpo_segments.json`. 

Navigate to the training directory and launch the predefined script for your chosen architecture:
```bash
cd training/LLaMA-Factory

# For Mistral 7B:
bash launch_dsa_dpo_training.sh

# For LLaMA 3.1 8B:
bash launch_dsa_dpo_llama3_training.sh

# For Gemma 2 9B:
bash launch_dsa_dpo_gemma2_training.sh
```

Trained models are exported to the `saves/` folder.
