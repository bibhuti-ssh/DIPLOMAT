"""
DSA-DPO Pipeline Runner
Orchestrates the complete negative session generation pipeline.
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm
import yaml

from session_generator import SessionGenerator, load_scenarios, load_scenarios_from_dir, GeneratedSession
from session_scorer import (
    SessionScorer,
    load_alignment_matrix,
    score_session_from_dialogue,
)
from llm_judge import LLMJudge, localize_error
from politeness_scorer import PolitenessScorer
from strategy_classifier import load_trained_gcns_classifier, GcNSClassifier


class DsaDpoPipeline:
    """
    Main pipeline for generating DSA-DPO training data.

    Steps:
    1. Generate dialogue sessions via self-play
    2. Score each session using the utility function
    3. Label sessions as positive/negative/borderline
    4. Localize errors for negative sessions
    5. Output DSA-DPO training data
    """

    def __init__(
        self,
        config_path: str = "dsa_dpo_pipeline/config.yaml",
        model_override: Optional[str] = None,
    ):
        self.config_path = config_path

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        print("Initializing pipeline components...")

        # Initialize GcNS classifier (CRITICAL FIX!)
        print("Loading GcNS strategy classifier...")
        gcns_model_path = "trained_models/gcns_negotiation_classifier"
        self.gcns_classifier = load_trained_gcns_classifier(model_path=gcns_model_path)
        print(
            f"✓ GcNS classifier loaded with {len(self.gcns_classifier.strategy_labels)} strategies"
        )

        # Initialize generator, scorer, judge with optional model override
        # Note: model_override applies to employer agent only; judge uses config default (gemini-2.5-pro)
        self.generator = SessionGenerator(config_path, model_override=model_override)
        self.scorer = SessionScorer(config_path, gcns_classifier=self.gcns_classifier)
        self.judge = LLMJudge(config_path)
        self.politeness_scorer = PolitenessScorer()

        # Load alignment matrices
        self.alignment_matrices = self._load_alignment_matrices()

        # Output config
        self.output_config = self.config["output"]
        os.makedirs(self.output_config["sessions_dir"], exist_ok=True)

    def _load_alignment_matrices(self) -> Dict:
        """Load negotiation and persuasion alignment matrices."""
        matrices = {}

        neg_path = "dsa_dpo_pipeline/alignment_matrices/negotiation_alignment.csv"
        if os.path.exists(neg_path):
            matrices["negotiation"] = load_alignment_matrix(neg_path)
            print(
                f"Loaded negotiation alignment matrix: {len(matrices['negotiation'])} pairs"
            )

        per_path = "dsa_dpo_pipeline/alignment_matrices/persuasion_alignment.csv"
        if os.path.exists(per_path):
            matrices["persuasion"] = load_alignment_matrix(per_path)
            print(
                f"Loaded persuasion alignment matrix: {len(matrices['persuasion'])} pairs"
            )

        return matrices

    def process_session(
        self,
        session: GeneratedSession,
        with_error_localization: bool = True,
    ) -> Dict:
        """
        Process a single generated session through the scoring pipeline.

        Args:
            session: Generated dialogue session
            with_error_localization: Whether to localize errors for negative sessions

        Returns:
            Complete session data with scores and labels
        """
        dialogue_dicts = [
            {
                "role": t.role,
                "content": t.content,
                "negotiation_strategy": t.negotiation_strategy,
                "persuasion_strategy": t.persuasion_strategy,
            }
            for t in session.dialogue
        ]

        # Get LLM judge scores (AQ, MS, CF)
        judge_result = self.judge.evaluate_session(
            dialogue=dialogue_dicts,
            negotiation_goal=session.scenario.get("negotiation_goal", ""),
            current_position=session.scenario.get("current_position", ""),
        )

        # Get politeness scores
        politeness_scores = self.politeness_scorer.compute_trajectory_scores(
            dialogue_dicts
        )

        # Add politeness to dialogue
        for i, turn_dict in enumerate(dialogue_dicts):
            scored = self.politeness_scorer.score_utterance(
                turn_dict["content"],
                turn_dict.get("persuasion_strategy"),
            )
            turn_dict["politeness_score"] = scored.score

        # Compute session score (CRITICAL FIX: pass gcns_classifier!)
        session_score = score_session_from_dialogue(
            session_id=session.session_id,
            dialogue=dialogue_dicts,
            scenario=session.scenario,
            llm_judge_results={
                "agreement_quality": judge_result.agreement_quality,
                "mutual_satisfaction": judge_result.mutual_satisfaction,
                "conflict_breakdown": judge_result.conflict_breakdown,
            },
            alignment_matrices=self.alignment_matrices,
            gcns_classifier=self.gcns_classifier,  # ADDED: Pass classifier
            config_path=self.config_path,
        )

        # Build result
        result = {
            "session_id": session.session_id,
            "scenario": session.scenario,
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
            "justifications": {
                "agreement_quality": judge_result.agreement_quality_justification,
                "mutual_satisfaction": judge_result.mutual_satisfaction_justification,
                "conflict_breakdown": judge_result.conflict_justification,
            },
        }

        print(f"\n[DEBUG] Session {session.session_id} Scores:")
        print(f"  - Agreement Quality: {judge_result.agreement_quality}")
        print(f"  - Mutual Satisfaction: {judge_result.mutual_satisfaction}")
        print(f"  - Conflict Breakdown: {judge_result.conflict_breakdown}")
        print(f"  - Strategy Alignment: {session_score.components.strategy_alignment}")
        print(f"  - Politeness Trajectory: {session_score.components.politeness_trajectory}")
        print(f"  - Total Score: {session_score.total_score}")

        # Error localization for negative sessions
        if with_error_localization and session_score.label == "negative":
            try:
                error_info = localize_error(
                    dialogue=dialogue_dicts,
                    scenario=session.scenario,
                    llm_judge=self.judge,
                )
                result["error_turn"] = error_info
            except Exception as e:
                print(f"Error localization failed: {e}")
                result["error_turn"] = {"index": -1, "reason": str(e)}

        return result

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file in the sessions directory."""
        import glob as _glob

        sessions_dir = self.output_config["sessions_dir"]
        pattern = os.path.join(sessions_dir, "checkpoint_*.json")
        checkpoints = _glob.glob(pattern)

        if not checkpoints:
            return None

        # Extract numbers and find the highest
        def extract_num(path):
            basename = os.path.basename(path)
            try:
                return int(basename.replace("checkpoint_", "").replace(".json", ""))
            except ValueError:
                return -1

        checkpoints.sort(key=extract_num)
        return checkpoints[-1]

    def run(
        self,
        scenarios: List[Dict],
        target_sessions: Optional[int] = None,
        save_interval: Optional[int] = None,
        resume: bool = False,
    ) -> Dict:
        """
        Run the complete pipeline.

        Args:
            scenarios: List of scenario configurations
            target_sessions: Number of sessions to generate (default from config)
            save_interval: Save progress every N sessions
            resume: If True, load from the latest checkpoint and continue

        Returns:
            Summary statistics
        """
        target = target_sessions or self.config["generation"]["target_sessions"]
        interval = save_interval or self.output_config["save_interval"]

        print(f"\n=== DSA-DPO Pipeline ===")
        print(f"Target sessions: {target}")
        print(f"Available scenarios: {len(scenarios)}")

        all_results = []
        start_index = 0
        stats = {
            "positive": 0,
            "negative": 0,
            "borderline": 0,
            "errors": 0,
        }

        # Resume from checkpoint if requested
        if resume:
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
                start_index = len(all_results)
                # Rebuild stats from loaded results
                for r in all_results:
                    label = r.get("label", "borderline")
                    if label in stats:
                        stats[label] += 1
                print(f"\n✓ Resuming from checkpoint: {checkpoint_path}")
                print(f"  Loaded {start_index} existing sessions")
                print(f"  Will generate {target - start_index} more sessions")
                print(f"  Stats so far: {stats}")
            else:
                print("\n⚠ --resume specified but no checkpoint found. Starting fresh.")

        if start_index >= target:
            print(f"\nAlready have {start_index} sessions (target={target}). Nothing to do.")
            self._save_results(all_results)
            self._print_summary(stats, len(all_results))
            return stats

        for i in tqdm(range(start_index, target), desc="Generating sessions", initial=start_index, total=target):
            try:
                # Sample scenario
                scenario = scenarios[i % len(scenarios)]

                # Generate session
                session = self.generator.generate_session(scenario)

                # Process and score
                result = self.process_session(session)
                all_results.append(result)

                # Update stats
                stats[result["label"]] += 1

                # Periodic save
                if (i + 1) % interval == 0:
                    self._save_checkpoint(all_results, i + 1)

            except Exception as e:
                print(f"Error processing session {i}: {e}")
                stats["errors"] += 1

        # Final save
        self._save_results(all_results)

        # Print summary
        self._print_summary(stats, len(all_results))

        return stats

    def _save_checkpoint(self, results: List[Dict], count: int):
        """Save intermediate checkpoint."""
        path = os.path.join(
            self.output_config["sessions_dir"], f"checkpoint_{count}.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nCheckpoint saved: {path}")

    def _save_results(self, results: List[Dict]):
        """Save final results and DSA-DPO training data."""
        # Save all sessions
        sessions_path = os.path.join(
            self.output_config["sessions_dir"], "all_sessions.json"
        )
        with open(sessions_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nAll sessions saved: {sessions_path}")

        # Generate DSA-DPO training format
        dsa_dpo_data = self._create_dsa_dpo_format(results)
        training_data_path = self.output_config["training_data_path"]
        os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
        with open(training_data_path, "w", encoding="utf-8") as f:
            json.dump(dsa_dpo_data, f, indent=2, ensure_ascii=False)
        print(f"DSA-DPO training data saved: {training_data_path} ({len(dsa_dpo_data)} pairs)")

    def _create_dsa_dpo_format(self, results: List[Dict]) -> List[Dict]:
        """
        Convert session results to DSA-DPO training format.

        For each negative session with error localization:
        - prompt: dialogue up to error turn
        - rejected: original error response
        - chosen: (to be generated or from positive sessions)
        """
        dsa_dpo_data = []

        for result in results:
            if result["label"] != "negative":
                continue

            error_turn = result.get("error_turn", {})
            error_index = error_turn.get("index", -1)

            if error_index < 0 or error_index >= len(result["dialogue"]):
                continue

            # Build prompt (dialogue up to error turn)
            dialogue_prefix = result["dialogue"][:error_index]
            prompt = self._format_dialogue_prompt(dialogue_prefix, result["scenario"])

            # Get rejected response
            rejected = result["dialogue"][error_index]["content"]

            dsa_dpo_entry = {
                "session_id": result["session_id"],
                "prompt": prompt,
                "rejected": rejected,
                "chosen": "",  # To be filled by response improvement or positive pairing
                "error_reason": error_turn.get("reason", ""),
                "scores": result["scores"],
            }
            dsa_dpo_data.append(dsa_dpo_entry)

        return dsa_dpo_data

    def _format_dialogue_prompt(self, dialogue: List[Dict], scenario: Dict) -> str:
        """Format dialogue prefix as prompt."""
        system = f"""You are an employer negotiating a job offer.
Goal: {scenario.get("negotiation_goal", "")}
Background: {scenario.get("background", "")}"""

        history = "\n".join(
            [f"{t['role'].capitalize()}: {t['content']}" for t in dialogue]
        )

        return f"{system}\n\n[Dialogue History]\n{history}\n\nGenerate the employer response:"

    def _print_summary(self, stats: Dict, total: int):
        """Print pipeline summary."""
        print("\n" + "=" * 50)
        print("DSA-DPO PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Total generated: {total}")
        print(
            f"  Positive: {stats['positive']} ({100 * stats['positive'] / max(total, 1):.1f}%)"
        )
        print(
            f"  Negative: {stats['negative']} ({100 * stats['negative'] / max(total, 1):.1f}%)"
        )
        print(
            f"  Borderline: {stats['borderline']} ({100 * stats['borderline'] / max(total, 1):.1f}%)"
        )
        print(f"  Errors: {stats['errors']}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Run DSA-DPO Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to conversations dataset JSON (legacy). Mutually exclusive with --scenario-dir.",
    )
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=None,
        help="Path to directory of scenario JSON files (e.g., hr_scenarios_json/). Preferred over --dataset.",
    )
    parser.add_argument(
        "--target", type=int, default=None, help="Override target sessions"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run with 5 sessions for testing"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model for employer agent and LLM judge (e.g., 'gemini-2.5-flash')",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint instead of starting fresh",
    )

    args = parser.parse_args()

    # Load scenarios
    if args.scenario_dir:
        print(f"Loading scenarios from directory: {args.scenario_dir}")
        scenarios = load_scenarios_from_dir(args.scenario_dir)
    elif args.dataset:
        print(f"Loading scenarios from dataset: {args.dataset}")
        scenarios = load_scenarios(args.dataset)
    else:
        # Default: try scenario dir first, then fall back to dataset
        default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hr_scenarios_json")
        default_dataset = "../hr_conversations/final_hr_conversations_dataset_fixed.json"
        if os.path.isdir(default_dir):
            print(f"Loading scenarios from directory: {default_dir}")
            scenarios = load_scenarios_from_dir(default_dir)
        else:
            print(f"Loading scenarios from dataset: {default_dataset}")
            scenarios = load_scenarios(default_dataset)
    print(f"Loaded {len(scenarios)} scenarios")

    # Initialize pipeline with optional model override
    if args.model:
        print(f"\n🔧 Using model override: {args.model}")
        print("   - Employer agent will use this model")
        print("   - LLM judge will use this model")

    pipeline = DsaDpoPipeline(args.config, model_override=args.model)

    # Run
    target = 3 if args.test else (args.target or None)
    pipeline.run(scenarios, target_sessions=target, resume=args.resume)


if __name__ == "__main__":
    main()
