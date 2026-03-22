#!/usr/bin/env python3
"""
DSA-DPO Pipeline Cost Calculator & Runner

This script:
1. Runs the 3-phase DSA-DPO pipeline for a single segment pair
2. Tracks all API costs (GPT-4o-mini, Mistral BC model)
3. Calculates cost per pair and projects cost for 1000 pairs

Usage:
    python calculate_dsa_dpo_cost.py

Models Used:
- Employer Agent (BC-trained): mistralai/Mistral-7B-Instruct-v0.3 with BC LoRA adapter
- Candidate Agent: GPT-4o-mini
- LLM Judge: GPT-4o-mini
- Error Localization: GPT-4o-mini
- Segment Extraction: GPT-4o-mini
"""

import os
import sys
import json
import yaml
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from session_generator import SessionGenerator, load_scenarios
from session_scorer import SessionScorer, load_alignment_matrix
from llm_judge import LLMJudge
from politeness_scorer import PolitenessScorer
from strategy_classifier import load_trained_gcns_classifier

# For Phase 2 and 3
from phase2_positive_sampling import Phase2PositiveSampler
from phase3_dsa_dpo_training import DsaDpoTrainingPrep


@dataclass
class CostTracker:
    """Track API costs throughout the pipeline."""
    
    # GPT-4o-mini costs (as of 2024)
    # Input: $0.150 / 1M tokens
    # Output: $0.600 / 1M tokens
    gpt4o_mini_input_cost_per_1m = 0.150
    gpt4o_mini_output_cost_per_1m = 0.600
    
    # Tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # Phase breakdown
    phase1_input_tokens: int = 0
    phase1_output_tokens: int = 0
    
    phase2_input_tokens: int = 0
    phase2_output_tokens: int = 0
    
    segment_input_tokens: int = 0
    segment_output_tokens: int = 0
    
    # Call counts
    candidate_calls: int = 0
    judge_calls: int = 0
    localization_calls: int = 0
    segment_extraction_calls: int = 0
    
    def add_usage(self, input_tokens: int, output_tokens: int, phase: str = None):
        """Add token usage."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        if phase == "phase1":
            self.phase1_input_tokens += input_tokens
            self.phase1_output_tokens += output_tokens
        elif phase == "phase2":
            self.phase2_input_tokens += input_tokens
            self.phase2_output_tokens += output_tokens
        elif phase == "segment":
            self.segment_input_tokens += input_tokens
            self.segment_output_tokens += output_tokens
    
    def calculate_cost(self) -> Dict[str, float]:
        """Calculate total cost and breakdown."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.gpt4o_mini_input_cost_per_1m
        output_cost = (self.total_output_tokens / 1_000_000) * self.gpt4o_mini_output_cost_per_1m
        total_cost = input_cost + output_cost
        
        # Phase breakdown
        phase1_input_cost = (self.phase1_input_tokens / 1_000_000) * self.gpt4o_mini_input_cost_per_1m
        phase1_output_cost = (self.phase1_output_tokens / 1_000_000) * self.gpt4o_mini_output_cost_per_1m
        phase1_cost = phase1_input_cost + phase1_output_cost
        
        phase2_input_cost = (self.phase2_input_tokens / 1_000_000) * self.gpt4o_mini_input_cost_per_1m
        phase2_output_cost = (self.phase2_output_tokens / 1_000_000) * self.gpt4o_mini_output_cost_per_1m
        phase2_cost = phase2_input_cost + phase2_output_cost
        
        segment_input_cost = (self.segment_input_tokens / 1_000_000) * self.gpt4o_mini_input_cost_per_1m
        segment_output_cost = (self.segment_output_tokens / 1_000_000) * self.gpt4o_mini_output_cost_per_1m
        segment_cost = segment_input_cost + segment_output_cost
        
        return {
            "total_cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "phase1_cost": phase1_cost,
            "phase2_cost": phase2_cost,
            "segment_cost": segment_cost,
            "cost_per_1000_pairs": total_cost * 1000,
        }
    
    def print_summary(self):
        """Print cost summary."""
        costs = self.calculate_cost()
        
        print("\n" + "="*80)
        print("💰 COST ANALYSIS")
        print("="*80)
        
        print(f"\n📊 Token Usage:")
        print(f"   Total Input Tokens:  {self.total_input_tokens:,}")
        print(f"   Total Output Tokens: {self.total_output_tokens:,}")
        print(f"   Total Tokens:        {self.total_input_tokens + self.total_output_tokens:,}")
        
        print(f"\n📞 API Call Breakdown:")
        print(f"   Candidate Agent Calls:      {self.candidate_calls}")
        print(f"   LLM Judge Calls:            {self.judge_calls}")
        print(f"   Error Localization Calls:   {self.localization_calls}")
        print(f"   Segment Extraction Calls:   {self.segment_extraction_calls}")
        
        print(f"\n💵 Cost Breakdown (GPT-4o-mini):")
        print(f"   Phase 1 (Negative Gen):     ${costs['phase1_cost']:.6f}")
        print(f"   Phase 2 (Positive Gen):     ${costs['phase2_cost']:.6f}")
        print(f"   Segment Extraction:         ${costs['segment_cost']:.6f}")
        print(f"   {'─'*40}")
        print(f"   Input Cost:                 ${costs['input_cost']:.6f}")
        print(f"   Output Cost:                ${costs['output_cost']:.6f}")
        print(f"   {'─'*40}")
        print(f"   TOTAL COST PER PAIR:        ${costs['total_cost']:.6f}")
        
        print(f"\n📈 Projections:")
        print(f"   Cost per 100 pairs:         ${costs['total_cost'] * 100:.2f}")
        print(f"   Cost per 1,000 pairs:       ${costs['cost_per_1000_pairs']:.2f}")
        print(f"   Cost per 10,000 pairs:      ${costs['cost_per_1000_pairs'] * 10:.2f}")
        
        print("\n" + "="*80)
        
        return costs


class DsaDpoCostCalculator:
    """
    Run DSA-DPO pipeline for cost calculation.
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        bc_model_path: str = "./trained_models/bc_lora_adapter",
    ):
        self.config_path = config_path
        self.bc_model_path = bc_model_path
        self.cost_tracker = CostTracker()
        
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Update config to use GPT-4o-mini and BC-trained Mistral
        self._update_config_for_cost_test()
        
        print("="*80)
        print("🧮 DSA-DPO PIPELINE COST CALCULATOR")
        print("="*80)
        print("\n📋 Configuration:")
        print(f"   Employer Agent:    Mistral-7B-v0.3 + BC LoRA")
        print(f"   BC Adapter Path:   {bc_model_path}")
        print(f"   Candidate Agent:   GPT-4o-mini")
        print(f"   LLM Judge:         GPT-4o-mini")
        print(f"   Error Localization: GPT-4o-mini")
        print(f"   Segment Extraction: GPT-4o-mini")
        
        # Initialize components
        print("\n⚙️  Initializing components...")
        self._init_components()
        
        # Output directory
        self.output_dir = "dsa_dpo_pipeline/outputs/cost_test"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "phase1"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "phase2"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "phase3"), exist_ok=True)
    
    def _update_config_for_cost_test(self):
        """Update config to use cost-effective models."""
        # Use GPT-4o-mini for all LLM calls
        self.config["llm_judge"]["model"] = "gpt-4o-mini"
        
        # Use BC-trained Mistral for employer
        self.config["self_play"]["employer_model"] = "mistralai/Mistral-7B-Instruct-v0.3"
        self.config["self_play"]["employer_adapter"] = self.bc_model_path
        
        # Use GPT-4o-mini for candidate
        self.config["self_play"]["candidate_model"] = "gpt-4o-mini"
        self.config["self_play"]["candidate_adapter"] = None
        
        # Reduce max turns for faster testing
        self.config["self_play"]["max_turns"] = 12
        self.config["self_play"]["min_turns"] = 6
    
    def _init_components(self):
        """Initialize pipeline components."""
        # Load GcNS classifier
        gcns_model_path = "trained_models/gcns_negotiation_classifier"
        self.gcns_classifier = load_trained_gcns_classifier(model_path=gcns_model_path)
        print(f"   ✓ GcNS classifier loaded")
        
        # Initialize generator (will use BC Mistral for employer)
        self.generator = SessionGenerator(self.config_path, model_override="gpt-4o-mini")
        print(f"   ✓ Session generator initialized")
        
        # Initialize scorer
        self.scorer = SessionScorer(self.config_path, gcns_classifier=self.gcns_classifier)
        print(f"   ✓ Session scorer initialized")
        
        # Initialize judge
        self.judge = LLMJudge(self.config_path, model_override="gpt-4o-mini")
        print(f"   ✓ LLM judge initialized")
        
        # Initialize politeness scorer
        self.politeness_scorer = PolitenessScorer()
        print(f"   ✓ Politeness scorer initialized")
        
        # Load alignment matrices
        self.alignment_matrices = self._load_alignment_matrices()
        print(f"   ✓ Alignment matrices loaded")
    
    def _load_alignment_matrices(self) -> Dict:
        """Load alignment matrices."""
        matrices = {}
        
        neg_path = "dsa_dpo_pipeline/alignment_matrices/negotiation_alignment.csv"
        if os.path.exists(neg_path):
            matrices["negotiation"] = load_alignment_matrix(neg_path)
        
        per_path = "dsa_dpo_pipeline/alignment_matrices/persuasion_alignment.csv"
        if os.path.exists(per_path):
            matrices["persuasion"] = load_alignment_matrix(per_path)
        
        return matrices
    
    def run_phase1_single_session(self, scenario: Dict) -> Optional[Dict]:
        """
        Phase 1: Generate a single negative session.
        """
        print("\n" + "="*80)
        print("📝 PHASE 1: Generating Negative Session")
        print("="*80)
        
        print(f"\n🎯 Scenario: {scenario.get('domain', 'Unknown')}")
        print(f"   Goal: {scenario.get('negotiation_goal', 'N/A')[:80]}...")
        
        start_time = time.time()
        
        try:
            # Generate session
            print("\n⏳ Generating dialogue...")
            session = self.generator.generate_session(scenario)
            
            # Convert to dict format
            dialogue_dicts = [
                {
                    "role": t.role,
                    "content": t.content,
                    "negotiation_strategy": t.negotiation_strategy,
                    "persuasion_strategy": t.persuasion_strategy,
                }
                for t in session.dialogue
            ]
            
            print(f"   ✓ Generated {len(dialogue_dicts)} turns")
            print(f"   ✓ Failure mode: {session.failure_mode}")
            
            # Score with LLM judge
            print("\n⏳ Scoring session with LLM judge...")
            judge_result = self.judge.evaluate_session(
                dialogue=dialogue_dicts,
                negotiation_goal=scenario.get("negotiation_goal", ""),
                current_position=scenario.get("current_position", ""),
            )
            
            # Track costs
            if hasattr(judge_result, 'usage'):
                self.cost_tracker.add_usage(
                    judge_result.usage.get("prompt_tokens", 0),
                    judge_result.usage.get("completion_tokens", 0),
                    phase="phase1"
                )
                self.cost_tracker.judge_calls += 1
            
            print(f"   ✓ AQ: {judge_result.agreement_quality:.2f}")
            print(f"   ✓ MS: {judge_result.mutual_satisfaction:.2f}")
            print(f"   ✓ CF: {judge_result.conflict_breakdown:.2f}")
            
            # Get politeness scores
            print("\n⏳ Computing politeness scores...")
            for turn_dict in dialogue_dicts:
                scored = self.politeness_scorer.score_utterance(
                    turn_dict["content"],
                    turn_dict.get("persuasion_strategy"),
                )
                turn_dict["politeness_score"] = scored.score
            
            # Compute session score
            from session_scorer import score_session_from_dialogue
            
            session_score = score_session_from_dialogue(
                session_id=session.session_id,
                dialogue=dialogue_dicts,
                scenario=scenario,
                llm_judge_results={
                    "agreement_quality": judge_result.agreement_quality,
                    "mutual_satisfaction": judge_result.mutual_satisfaction,
                    "conflict_breakdown": judge_result.conflict_breakdown,
                },
                alignment_matrices=self.alignment_matrices,
                gcns_classifier=self.gcns_classifier,
                config_path=self.config_path,
            )
            
            print(f"   ✓ Total Score: {session_score.total_score:.3f}")
            print(f"   ✓ Label: {session_score.label}")
            
            # Error localization if negative
            error_turn = None
            if session_score.label == "negative":
                print("\n⏳ Localizing error turn...")
                from llm_judge import localize_error
                
                error_turn = localize_error(
                    dialogue=dialogue_dicts,
                    scenario=scenario,
                    llm_judge=self.judge,
                )
                
                # Track cost (estimate since localize_error doesn't return usage)
                if error_turn:
                    # Estimate: scenario + dialogue + prompt ≈ 1500 tokens input, 150 tokens output
                    estimated_input = 1500
                    estimated_output = 150
                    self.cost_tracker.add_usage(estimated_input, estimated_output, phase="phase1")
                    self.cost_tracker.localization_calls += 1
                    
                    print(f"   ✓ Error at turn {error_turn['index']}")
                    print(f"   ✓ Reason: {error_turn['reason'][:100]}...")
            
            elapsed = time.time() - start_time
            print(f"\n⏱️  Phase 1 completed in {elapsed:.1f}s")
            
            # Build result
            result = {
                "session_id": session.session_id,
                "scenario": scenario,
                "dialogue": dialogue_dicts,
                "failure_mode": session.failure_mode,
                "scores": {
                    "agreement_quality": judge_result.agreement_quality,
                    "mutual_satisfaction": judge_result.mutual_satisfaction,
                    "strategy_alignment": session_score.components.strategy_alignment,
                    "politeness_trajectory": session_score.components.politeness_trajectory,
                    "conflict_breakdown": judge_result.conflict_breakdown,
                    "total_score": session_score.total_score,
                },
                "label": session_score.label,
                "error_turn": error_turn,
            }
            
            # Save
            output_path = os.path.join(self.output_dir, "phase1", "negative_session.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"\n💾 Saved to: {output_path}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error in Phase 1: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_phase2_positive_sampling(self, negative_session: Dict) -> Optional[Dict]:
        """
        Phase 2: Generate positive counterpart.
        """
        print("\n" + "="*80)
        print("✨ PHASE 2: Generating Positive Counterpart")
        print("="*80)
        
        if negative_session["label"] != "negative" or not negative_session.get("error_turn"):
            print("\n⚠️  Session is not negative or has no error turn. Skipping Phase 2.")
            return None
        
        error_turn_index = negative_session["error_turn"]["index"]
        error_reason = negative_session["error_turn"]["reason"]
        
        print(f"\n🎯 Error Turn: {error_turn_index}")
        print(f"   Reason: {error_reason[:100]}...")
        
        start_time = time.time()
        
        try:
            # Initialize Phase 2 sampler
            sampler = Phase2PositiveSampler(
                config_path=self.config_path,
                model_override="gpt-4o-mini",
                num_candidates=3,
            )
            
            # Process the negative session
            print(f"\n⏳ Generating 3 positive candidates...")
            dsa_dpo_pair = sampler.process_negative_session(negative_session)
            
            if not dsa_dpo_pair:
                print("\n⚠️  Failed to generate positive counterpart")
                return None
            
            # Track costs (approximate from candidate generation)
            # Each candidate involves: 3 continuations × ~2 turns each × (candidate + judge)
            # Rough estimate: 3 candidates × 4 LLM calls × ~1000 tokens
            estimated_input_tokens = 3 * 4 * 800
            estimated_output_tokens = 3 * 4 * 200
            self.cost_tracker.add_usage(estimated_input_tokens, estimated_output_tokens, phase="phase2")
            self.cost_tracker.candidate_calls += 12  # 3 candidates × ~4 turns
            self.cost_tracker.judge_calls += 3  # 3 judge calls
            
            print(f"\n✓ Generated positive counterpart:")
            print(f"   Positive Score: {dsa_dpo_pair.positive_total_score:.3f}")
            print(f"   Negative Score: {dsa_dpo_pair.negative_total_score:.3f}")
            print(f"   Improvement: +{dsa_dpo_pair.positive_total_score - dsa_dpo_pair.negative_total_score:.3f}")
            
            elapsed = time.time() - start_time
            print(f"\n⏱️  Phase 2 completed in {elapsed:.1f}s")
            
            # Save
            output_path = os.path.join(self.output_dir, "phase2", "dsa_dpo_pair.json")
            with open(output_path, "w") as f:
                json.dump(asdict(dsa_dpo_pair), f, indent=2, default=str)
            
            print(f"\n💾 Saved to: {output_path}")
            
            # Extract segment
            print("\n⏳ Extracting key differentiating segment...")
            segment_pair = sampler.extract_key_segment(
                scenario=negative_session["scenario"],
                negative_dialogue=negative_session["dialogue"],
                positive_dialogue=[
                    {"role": t["role"], "content": t["content"],
                     "negotiation_strategy": t.get("negotiation_strategy", ""),
                     "persuasion_strategy": t.get("persuasion_strategy", "")}
                    for t in (dsa_dpo_pair.shared_context + dsa_dpo_pair.positive_continuation)
                ],
                error_turn_index=error_turn_index,
            )
            
            # Track segment extraction cost
            if segment_pair:
                estimated_segment_input = 1500  # Prompt with both dialogues
                estimated_segment_output = 100  # JSON response
                self.cost_tracker.add_usage(estimated_segment_input, estimated_segment_output, phase="segment")
                self.cost_tracker.segment_extraction_calls += 1
                
                print(f"   ✓ Segment: turns {segment_pair.start_index} to {segment_pair.end_index}")
                print(f"   ✓ Reason: {segment_pair.selection_reason[:100]}...")
                
                # Save segment
                segment_output = os.path.join(self.output_dir, "phase2", "segment_pair.json")
                with open(segment_output, "w") as f:
                    json.dump(asdict(segment_pair), f, indent=2, default=str)
                
                print(f"\n💾 Segment saved to: {segment_output}")
            else:
                print("\n⚠️  Failed to extract segment")
            
            return {
                "dsa_dpo_pair": dsa_dpo_pair,
                "segment_pair": segment_pair,
            }
            
        except Exception as e:
            print(f"\n❌ Error in Phase 2: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_phase3_training_prep(self, phase2_result: Dict) -> bool:
        """
        Phase 3: Prepare training data.
        """
        print("\n" + "="*80)
        print("🎓 PHASE 3: Training Data Preparation")
        print("="*80)
        
        if not phase2_result or not phase2_result.get("segment_pair"):
            print("\n⚠️  No segment pair available. Skipping Phase 3.")
            return False
        
        try:
            segment_pair = phase2_result["segment_pair"]
            
            # Convert to DSA-DPO format
            print("\n⏳ Converting to DSA-DPO training format...")
            
            context_before = segment_pair.context_before
            positive_segment = segment_pair.positive_segment
            negative_segment = segment_pair.negative_segment
            
            # Build conversation context
            conversations = []
            for turn in context_before:
                role = turn["role"]
                content = turn["content"]
                
                if role == "candidate":
                    conversations.append({"from": "human", "value": content})
                elif role == "employer":
                    conversations.append({"from": "gpt", "value": content})
            
            # Find first employer turn in segments
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
                print("\n⚠️  No employer turns found in segments")
                return False
            
            # Add candidate turns before employer response to context
            for i in range(first_employer_idx_pos):
                if positive_segment[i]["role"] == "candidate":
                    conversations.append({"from": "human", "value": positive_segment[i]["content"]})
            
            # Get employer responses
            positive_response = positive_segment[first_employer_idx_pos]["content"]
            negative_response = negative_segment[first_employer_idx_neg]["content"]
            
            # Create DSA-DPO example
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
                "_metadata": {
                    "segment_id": segment_pair.segment_id,
                    "selection_reason": segment_pair.selection_reason,
                }
            }
            
            print(f"   ✓ Context turns: {len(conversations)}")
            print(f"   ✓ Chosen response: {len(positive_response)} chars")
            print(f"   ✓ Rejected response: {len(negative_response)} chars")
            
            # Save
            output_path = os.path.join(self.output_dir, "phase3", "dsa_dpo_training_example.json")
            with open(output_path, "w") as f:
                json.dump([dsa_dpo_example], f, indent=2)
            
            print(f"\n💾 Saved to: {output_path}")
            print(f"\n✓ DSA-DPO training example ready!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error in Phase 3: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self, scenario: Dict):
        """Run complete 3-phase pipeline for cost calculation."""
        print("\n" + "="*80)
        print("🚀 STARTING 3-PHASE DSA-DPO PIPELINE")
        print("="*80)
        
        overall_start = time.time()
        
        # Phase 1
        negative_session = self.run_phase1_single_session(scenario)
        if not negative_session:
            print("\n❌ Phase 1 failed")
            return
        
        # Phase 2
        phase2_result = self.run_phase2_positive_sampling(negative_session)
        if not phase2_result:
            print("\n❌ Phase 2 failed")
            return
        
        # Phase 3
        success = self.run_phase3_training_prep(phase2_result)
        if not success:
            print("\n❌ Phase 3 failed")
            return
        
        overall_elapsed = time.time() - overall_start
        
        # Print final summary
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE")
        print("="*80)
        print(f"\n⏱️  Total Time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
        
        # Print cost summary
        costs = self.cost_tracker.print_summary()
        
        # Save cost report
        cost_report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "employer_agent": "mistralai/Mistral-7B-Instruct-v0.3 + BC LoRA",
                "candidate_agent": "GPT-4o-mini",
                "llm_judge": "GPT-4o-mini",
                "error_localization": "GPT-4o-mini",
                "segment_extraction": "GPT-4o-mini",
            },
            "token_usage": {
                "total_input_tokens": self.cost_tracker.total_input_tokens,
                "total_output_tokens": self.cost_tracker.total_output_tokens,
                "phase1_input_tokens": self.cost_tracker.phase1_input_tokens,
                "phase1_output_tokens": self.cost_tracker.phase1_output_tokens,
                "phase2_input_tokens": self.cost_tracker.phase2_input_tokens,
                "phase2_output_tokens": self.cost_tracker.phase2_output_tokens,
                "segment_input_tokens": self.cost_tracker.segment_input_tokens,
                "segment_output_tokens": self.cost_tracker.segment_output_tokens,
            },
            "api_calls": {
                "candidate_calls": self.cost_tracker.candidate_calls,
                "judge_calls": self.cost_tracker.judge_calls,
                "localization_calls": self.cost_tracker.localization_calls,
                "segment_extraction_calls": self.cost_tracker.segment_extraction_calls,
            },
            "costs": costs,
            "elapsed_time_seconds": overall_elapsed,
        }
        
        cost_report_path = os.path.join(self.output_dir, "cost_report.json")
        with open(cost_report_path, "w") as f:
            json.dump(cost_report, f, indent=2)
        
        print(f"\n💾 Cost report saved to: {cost_report_path}")
        
        return cost_report


def main():
    parser = argparse.ArgumentParser(description="Calculate DSA-DPO pipeline cost per segment pair")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--bc-adapter",
        type=str,
        default="./trained_models/bc_lora_adapter",
        help="Path to BC-trained Mistral LoRA adapter"
    )
    parser.add_argument(
        "--scenario-file",
        type=str,
        default="./data/scenarios.json",
        help="Path to scenarios file"
    )
    
    args = parser.parse_args()
    
    # Check if BC adapter exists
    if not os.path.exists(args.bc_adapter):
        print(f"\n❌ BC adapter not found at: {args.bc_adapter}")
        print("   Please train the BC model first.")
        sys.exit(1)
    
    # Load a sample scenario
    print(f"\n📂 Loading scenarios from: {args.scenario_file}")
    scenarios = load_scenarios(args.scenario_file)
    
    if not scenarios:
        print("❌ No scenarios loaded")
        sys.exit(1)
    
    # Pick first scenario
    scenario = scenarios[0]
    print(f"   ✓ Loaded {len(scenarios)} scenarios")
    print(f"   ✓ Using scenario: {scenario.get('domain', 'Unknown')}")
    
    # Initialize calculator
    calculator = DsaDpoCostCalculator(
        config_path=args.config,
        bc_model_path=args.bc_adapter,
    )
    
    # Run pipeline
    cost_report = calculator.run(scenario)
    
    if cost_report:
        print("\n" + "="*80)
        print("📊 FINAL SUMMARY")
        print("="*80)
        print(f"\n✅ Successfully generated 1 DSA-DPO segment pair")
        print(f"\n💰 Cost per pair: ${cost_report['costs']['total_cost']:.6f}")
        print(f"💰 Cost per 1,000 pairs: ${cost_report['costs']['cost_per_1000_pairs']:.2f}")
        print(f"\n⏱️  Time per pair: {cost_report['elapsed_time_seconds']:.1f}s")
        print(f"⏱️  Estimated time for 1,000 pairs: {cost_report['elapsed_time_seconds'] * 1000 / 3600:.1f} hours")
        
        print("\n" + "="*80)
        print("✅ COST CALCULATION COMPLETE")
        print("="*80)


if __name__ == "__main__":
    main()
