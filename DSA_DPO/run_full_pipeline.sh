#!/usr/bin/env bash
# =============================================================================
# DSA-DPO Full Pipeline Runner
# Generates ~1000 negative sessions (Phase 1), positive counterparts (Phase 2),
# and prepares DSA-DPO training data (Phase 3).
#
# Candidate model: BC-finetuned Mistral-7B-Instruct-v0.3
# Employer agent:  Gemini 2.5 Flash (via API)
# LLM Judge:       Gemini 2.5 Pro  (via API)
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   nohup ./run_full_pipeline.sh > pipeline_master.log 2>&1 &
#   # or
#   screen -S diplomat ./run_full_pipeline.sh
#
# Author: auto-generated
# =============================================================================

set -euo pipefail   # exit on error, undefined vars, pipe failures

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

# Paths (relative to repo root)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINE_DIR="${REPO_ROOT}/dsa_dpo_pipeline"
CONFIG_PATH="${PIPELINE_DIR}/config.yaml"
SCENARIO_DIR="${REPO_ROOT}/hr_scenarios_json"
# Legacy dataset (no longer used by default):
# DATASET_PATH="${REPO_ROOT}/hr_conversations/final_hr_conversations_dataset_fixed.json"

# Model override for employer agent in Phase 1 (judge uses config default)
MODEL_OVERRIDE="gemini-2.5-flash"

# Restrict to a single GPU
export CUDA_VISIBLE_DEVICES=0

# Phase 1 - Negative session generation
TARGET_SESSIONS=800
SAVE_INTERVAL=50

# Phase 2 - Positive counterpart sampling
NUM_CANDIDATES=2   # generate 2 positive candidates per negative, pick best

# Logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PIPELINE_DIR}/logs"
PHASE1_LOG="${LOG_DIR}/phase1_${TIMESTAMP}.log"
PHASE2_LOG="${LOG_DIR}/phase2_${TIMESTAMP}.log"
PHASE3_LOG="${LOG_DIR}/phase3_${TIMESTAMP}.log"
SUMMARY_LOG="${LOG_DIR}/pipeline_summary_${TIMESTAMP}.log"

# ─────────────────────────── HELPER FUNCTIONS ────────────────────────────────

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "${SUMMARY_LOG}"
}

die() {
    log "FATAL: $*"
    log "Pipeline aborted. Check logs in ${LOG_DIR}/"
    exit 1
}

elapsed() {
    local start=$1
    local end=$(date +%s)
    local secs=$((end - start))
    printf '%02dh:%02dm:%02ds' $((secs/3600)) $((secs%3600/60)) $((secs%60))
}

# ─────────────────────────── PRE-FLIGHT CHECKS ──────────────────────────────

preflight() {
    log "========== PRE-FLIGHT CHECKS =========="

    # 1. Python
    command -v python3 &>/dev/null || die "python3 not found in PATH"
    log "Python: $(python3 --version 2>&1)"

    # 2. GEMINI_API_KEY
    if [[ -z "${GEMINI_API_KEY:-}" ]]; then
        die "GEMINI_API_KEY environment variable is not set. Export it before running."
    fi
    log "GEMINI_API_KEY: set (${#GEMINI_API_KEY} chars)"

    # 3. Scenario directory
    if [[ ! -d "${SCENARIO_DIR}" ]]; then
        die "Scenario directory not found: ${SCENARIO_DIR}"
    fi
    SCENARIO_COUNT=$(python3 -c "
import json, os, glob
total = 0
for f in glob.glob(os.path.join('${SCENARIO_DIR}', '*.json')):
    d = json.load(open(f))
    total += len(d) if isinstance(d, list) else 1
print(total)
" 2>/dev/null || echo "?")
    log "Scenario dir: ${SCENARIO_DIR} (${SCENARIO_COUNT} scenarios)"

    # 4. Config file
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        die "Config not found: ${CONFIG_PATH}"
    fi
    log "Config: ${CONFIG_PATH}"

    # 5. GcNS trained model
    local gcns_dir="${PIPELINE_DIR}/trained_models/gcns_negotiation_classifier"
    if [[ ! -f "${gcns_dir}/model.safetensors" ]]; then
        die "GcNS classifier weights not found: ${gcns_dir}/model.safetensors"
    fi
    log "GcNS classifier: ${gcns_dir} ✓"

    # 6. BC adapter (Mistral)
    local adapter_dir="${REPO_ROOT}/bc_finetuning/outputs/mistral_bc_lora/final_adapter"
    if [[ ! -d "${adapter_dir}" ]]; then
        log "WARNING: Mistral BC adapter not found at ${adapter_dir}"
        log "         Pipeline will use mock candidate responses."
    else
        log "Mistral BC adapter: ${adapter_dir} ✓"
    fi

    # 7. Alignment matrices
    for mat in negotiation_alignment.csv persuasion_alignment.csv; do
        if [[ ! -f "${PIPELINE_DIR}/alignment_matrices/${mat}" ]]; then
            die "Alignment matrix not found: ${PIPELINE_DIR}/alignment_matrices/${mat}"
        fi
    done
    log "Alignment matrices: ✓"

    # 8. Python imports quick check
    python3 -c "
import yaml, tqdm, torch, transformers, peft
print('Core packages OK')
" 2>/dev/null || die "Missing Python dependencies. Run: pip install -r ${PIPELINE_DIR}/requirements.txt"
    log "Python dependencies: ✓"

    # 9. GPU check (informational)
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
        GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')" 2>/dev/null || echo "unknown")
        log "GPU: ${GPU_NAME} (${GPU_MEM})"
    else
        log "WARNING: No CUDA GPU detected. BC model will be slow on CPU."
    fi

    log "========== PRE-FLIGHT PASSED =========="
    echo ""
}

# ─────────────────────────── PHASE 1 ────────────────────────────────────────

run_phase1() {
    local start_time=$(date +%s)
    log ""
    log "╔══════════════════════════════════════════════════════════════════╗"
    log "║  PHASE 1: Negative Session Generation (target=${TARGET_SESSIONS})          ║"
    log "╚══════════════════════════════════════════════════════════════════╝"
    log "Log → ${PHASE1_LOG}"
    log ""

    cd "${PIPELINE_DIR}"

    local sessions_file="${PIPELINE_DIR}/outputs/sessions/all_sessions.json"

    # If final Phase 1 output already exists, skip regeneration
    if [[ -f "${sessions_file}" ]]; then
        local existing_count=$(python3 -c "import json; print(len(json.load(open('${sessions_file}'))))" 2>/dev/null || echo 0)
        if [[ "${existing_count}" -gt 0 ]]; then
            log "Found existing Phase 1 output: ${sessions_file} (${existing_count} sessions)"
            log "Skipping Phase 1 generation and proceeding to Phase 2."
            return 0
        fi
    fi

    # Check for existing checkpoints to auto-resume
    local resume_flag=""
    local latest_ckpt=$(find "${PIPELINE_DIR}/outputs/sessions" -name 'checkpoint_*.json' 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [[ -n "${latest_ckpt}" ]]; then
        local ckpt_count=$(python3 -c "import json; print(len(json.load(open('${latest_ckpt}'))))" 2>/dev/null || echo 0)
        log "Found existing checkpoint: ${latest_ckpt} (${ckpt_count} sessions)"
        log "Resuming from checkpoint..."
        resume_flag="--resume"
    fi

    python3 phase1_negative_generation.py \
        --config "${CONFIG_PATH}" \
        --scenario-dir "${SCENARIO_DIR}" \
        --target "${TARGET_SESSIONS}" \
        --model "${MODEL_OVERRIDE}" \
        ${resume_flag} \
        2>&1 | tee "${PHASE1_LOG}"

    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -ne 0 ]]; then
        die "Phase 1 failed with exit code ${exit_code}. See ${PHASE1_LOG}"
    fi

    # Validate output
    if [[ ! -f "${sessions_file}" ]]; then
        die "Phase 1 completed but output file missing: ${sessions_file}"
    fi

    local total=$(python3 -c "import json; d=json.load(open('${sessions_file}')); print(len(d))")
    local neg=$(python3 -c "import json; d=json.load(open('${sessions_file}')); print(sum(1 for s in d if s.get('label')=='negative'))")
    local pos=$(python3 -c "import json; d=json.load(open('${sessions_file}')); print(sum(1 for s in d if s.get('label')=='positive'))")
    local with_err=$(python3 -c "import json; d=json.load(open('${sessions_file}')); print(sum(1 for s in d if s.get('label')=='negative' and s.get('error_turn',{}).get('index',-1)>=0))")

    log ""
    log "Phase 1 Results:"
    log "  Total sessions:        ${total}"
    log "  Positive:              ${pos}"
    log "  Negative:              ${neg}"
    log "  Negative w/ error loc: ${with_err}"
    log "  Elapsed:               $(elapsed $start_time)"
    log ""

    if [[ "${with_err}" -eq 0 ]]; then
        die "Phase 1 produced 0 negative sessions with error localization. Cannot proceed to Phase 2."
    fi

    log "Phase 1 COMPLETE ✓"
}

# ─────────────────────────── PHASE 2 ────────────────────────────────────────

run_phase2() {
    local start_time=$(date +%s)
    log ""
    log "╔══════════════════════════════════════════════════════════════════╗"
    log "║  PHASE 2: Positive Counterpart Sampling (${NUM_CANDIDATES} candidates each) ║"
    log "╚══════════════════════════════════════════════════════════════════╝"
    log "Log → ${PHASE2_LOG}"
    log ""

    cd "${PIPELINE_DIR}"

    local sessions_file="${PIPELINE_DIR}/outputs/sessions/all_sessions.json"

    python3 phase2_positive_sampling.py \
        --input "${sessions_file}" \
        --config "${CONFIG_PATH}" \
        --num-candidates "${NUM_CANDIDATES}" \
        2>&1 | tee "${PHASE2_LOG}"

    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -ne 0 ]]; then
        die "Phase 2 failed with exit code ${exit_code}. See ${PHASE2_LOG}"
    fi

    # Validate output
    local pairs_file="${PIPELINE_DIR}/outputs/phase2_output/dsa_dpo_pairs.json"
    local segments_file="${PIPELINE_DIR}/outputs/phase2_output/dsa_dpo_segment_pairs.json"

    if [[ ! -f "${pairs_file}" ]]; then
        die "Phase 2 completed but pairs file missing: ${pairs_file}"
    fi

    local pairs_count=$(python3 -c "import json; print(len(json.load(open('${pairs_file}'))))")
    local segments_count=0
    if [[ -f "${segments_file}" ]]; then
        segments_count=$(python3 -c "import json; print(len(json.load(open('${segments_file}'))))")
    fi

    local avg_improvement="N/A"
    if [[ "${pairs_count}" -gt 0 ]]; then
        avg_improvement=$(python3 -c "
import json
pairs = json.load(open('${pairs_file}'))
imp = [p.get('score_improvement',0) for p in pairs]
print(f'{sum(imp)/len(imp):.3f}' if imp else 'N/A')
")
    fi

    log ""
    log "Phase 2 Results:"
    log "  DSA-DPO pairs created:       ${pairs_count}"
    log "  Segments extracted:       ${segments_count}"
    log "  Avg score improvement:    ${avg_improvement}"
    log "  Elapsed:                  $(elapsed $start_time)"
    log ""

    if [[ "${segments_count}" -eq 0 ]]; then
        log "WARNING: No segments extracted. Phase 3 will use pair-level data."
    fi

    log "Phase 2 COMPLETE ✓"
}

# ─────────────────────────── PHASE 3 ────────────────────────────────────────

run_phase3() {
    local start_time=$(date +%s)
    log ""
    log "╔══════════════════════════════════════════════════════════════════╗"
    log "║  PHASE 3: DSA-DPO Training Data Preparation                      ║"
    log "╚══════════════════════════════════════════════════════════════════╝"
    log "Log → ${PHASE3_LOG}"
    log ""

    cd "${PIPELINE_DIR}"

    local segments_file="${PIPELINE_DIR}/outputs/phase2_output/dsa_dpo_segment_pairs.json"

    # Check if LLaMA-Factory directory exists; if not, create data output locally
    local output_dir="${PIPELINE_DIR}/outputs/dsa_dpo_training_data"
    if [[ -d "${REPO_ROOT}/training/LLaMA-Factory/data" ]]; then
        output_dir="${REPO_ROOT}/training/LLaMA-Factory/data"
    fi

    python3 phase3_dsa_dpo_training.py \
        --segment-pairs "${segments_file}" \
        --output-dir "${output_dir}" \
        --dataset-name "hr_dsa_dpo_segments" \
        2>&1 | tee "${PHASE3_LOG}"

    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -ne 0 ]]; then
        die "Phase 3 failed with exit code ${exit_code}. See ${PHASE3_LOG}"
    fi

    # Count training examples
    local training_file="${output_dir}/hr_dsa_dpo_segments.json"
    local train_count=0
    if [[ -f "${training_file}" ]]; then
        train_count=$(python3 -c "import json; print(len(json.load(open('${training_file}'))))")
    fi

    log ""
    log "Phase 3 Results:"
    log "  Training examples:   ${train_count}"
    log "  Output:              ${training_file}"
    log "  Elapsed:             $(elapsed $start_time)"
    log ""
    log "Phase 3 COMPLETE ✓"
}

# ─────────────────────────── FINAL SUMMARY ──────────────────────────────────

print_final_summary() {
    local total_elapsed=$(elapsed $PIPELINE_START)

    log ""
    log "╔══════════════════════════════════════════════════════════════════╗"
    log "║  ALL PHASES COMPLETE                                           ║"
    log "╚══════════════════════════════════════════════════════════════════╝"
    log ""
    log "Total wall-time: ${total_elapsed}"
    log ""
    log "Output artifacts:"
    log "  Phase 1 sessions:     ${PIPELINE_DIR}/outputs/sessions/all_sessions.json"
    log "  Phase 2 pairs:        ${PIPELINE_DIR}/outputs/phase2_output/dsa_dpo_pairs.json"
    log "  Phase 2 segments:     ${PIPELINE_DIR}/outputs/phase2_output/dsa_dpo_segment_pairs.json"
    log "  Phase 3 training:     (see Phase 3 log for exact path)"
    log ""
    log "Logs:"
    log "  Phase 1:  ${PHASE1_LOG}"
    log "  Phase 2:  ${PHASE2_LOG}"
    log "  Phase 3:  ${PHASE3_LOG}"
    log "  Summary:  ${SUMMARY_LOG}"
    log ""
    log "Next step: Run DSA-DPO training with LLaMA-Factory."
    log "  cd training/LLaMA-Factory"
    log "  llamafactory-cli train examples/train_lora/hr_dsa_dpo_training.yaml"
    log ""
    log "Pipeline finished successfully at $(date '+%Y-%m-%d %H:%M:%S')."
}

# ─────────────────────────── MAIN ───────────────────────────────────────────

main() {
    PIPELINE_START=$(date +%s)

    # Create log directory
    mkdir -p "${LOG_DIR}"
    touch "${SUMMARY_LOG}"

    log "================================================================"
    log "  DSA-DPO FULL PIPELINE"
    log "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    log "  Target: ${TARGET_SESSIONS} sessions"
    log "  Candidate: Mistral-7B-Instruct BC-finetuned"
    log "  Employer/Judge: ${MODEL_OVERRIDE}"
    log "================================================================"

    # Run everything
    preflight
    run_phase1
    run_phase2
    run_phase3
    print_final_summary
}

main "$@"
