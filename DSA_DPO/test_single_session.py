"""
Script to run a single session generation and scoring for debugging.
Run with: python3 test_single_session.py --model gemini-2.5-pro
"""

import argparse
import json
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from session_generator import SessionGenerator, load_scenarios
from session_scorer import SessionScorer, score_session_from_dialogue
from llm_judge import LLMJudge
from politeness_scorer import PolitenessScorer
from strategy_classifier import load_trained_gcns_classifier

def main():
    parser = argparse.ArgumentParser(description="Test Single Session Generation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--output", type=str, default="debug_session3.json")
    args = parser.parse_args()

    print(f"\n🔧 Testing Single Session with Model: {args.model}")

    # 1. Load Scenarios
    print("Loading scenarios...")
    scenarios = load_scenarios("../hr_conversations/final_hr_conversations_dataset_fixed.json")
    scenario = scenarios[0] # Pick first one
    print(f"Loaded scenario: {scenario.get('scenario_id', 'Unknown')}")

    # 2. Initialize Components
    print("\nInitializing components...")
    
    # GcNS
    gcns_model_path = "trained_models/gcns_negotiation_classifier"
    gcns_classifier = load_trained_gcns_classifier(model_path=gcns_model_path)
    print("✓ GcNS classifier loaded")

    # Generator
    generator = SessionGenerator(args.config, model_override=args.model)
    print("✓ Generator initialized")

    # Judge
    judge = LLMJudge(args.config, model_override=args.model)
    print("✓ Judge initialized")

    # Politeness
    politeness_scorer = PolitenessScorer(use_convokit=True)
    print("✓ Politeness scorer initialized")

    # 3. Generate Session
    print("\n🚀 Generating session...")
    session = generator.generate_session(scenario)
    print(f"Session generated with {len(session.dialogue)} turns.")

    # 4. Score Session
    print("\n⚖️  Scoring session...")
    
    dialogue_dicts = [
        {
            "role": t.role,
            "content": t.content,
            "negotiation_strategy": t.negotiation_strategy,
            "persuasion_strategy": t.persuasion_strategy,
        }
        for t in session.dialogue
    ]

    # Judge Evaluation
    judge_result = judge.evaluate_session(
        dialogue=dialogue_dicts,
        negotiation_goal=session.scenario.get("negotiation_goal", ""),
        current_position=session.scenario.get("current_position", ""),
    )
    print("✓ LLM Judge evaluation complete")

    # Politeness 
    for i, turn_dict in enumerate(dialogue_dicts):
        scored = politeness_scorer.score_utterance(
            turn_dict["content"],
            turn_dict.get("persuasion_strategy"),
        )
        turn_dict["politeness_score"] = scored.score
    print("✓ Politeness scoring complete")

    # Final Session Score
    session_score = score_session_from_dialogue(
        session_id=session.session_id,
        dialogue=dialogue_dicts,
        scenario=session.scenario,
        llm_judge_results={
            "agreement_quality": judge_result.agreement_quality,
            "mutual_satisfaction": judge_result.mutual_satisfaction,
            "conflict_breakdown": judge_result.conflict_breakdown,
        },
        alignment_matrices={}, # Skip for debug
        gcns_classifier=gcns_classifier,
        config_path=args.config,
    )

    # 5. Print Results
    print("\n" + "="*50)
    print(f"SESSION RESULTS ({session.session_id})")
    print("="*50)
    
    print(f"\n[SCORES]")
    print(f"Agreement Quality:   {judge_result.agreement_quality}")
    print(f"Mutual Satisfaction: {judge_result.mutual_satisfaction}")
    print(f"Conflict Breakdown:  {judge_result.conflict_breakdown}")
    print(f"Strategy Alignment:  {session_score.components.strategy_alignment}")
    print(f"Politeness Score:    {session_score.components.politeness_trajectory}")
    print(f"TOTAL SCORE:         {session_score.total_score:.4f}")
    
    print(f"\n[LABEL]: {session_score.label.upper()}")

    print(f"\n[JUSTIFICATIONS]")
    print(f"Agreement: {judge_result.agreement_quality_justification}")
    print(f"Satisfaction: {judge_result.mutual_satisfaction_justification}")
    print(f"Conflict: {judge_result.conflict_justification}")

    # Save prompt if judge output was empty (handled continuously in code, but good to note)
    if os.path.exists("debug_failed_prompt.txt"):
        print("\n[WARNING] debug_failed_prompt.txt was created during this run!")

    # Save result
    result = {
        "session_id": session.session_id,
        "dialogue": dialogue_dicts,
        "scores": {
            "agreement_quality": judge_result.agreement_quality,
            "mutual_satisfaction": judge_result.mutual_satisfaction,
            "conflict_breakdown": judge_result.conflict_breakdown,
            "total_score": session_score.total_score,
        }
    }
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved full result to {args.output}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
