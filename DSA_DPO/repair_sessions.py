"""
Repair Script for all_sessions.json

Finds negative sessions with failed error localization (index=-1)
and re-runs the LLM judge error localization call with retries.

Usage:
    cd dsa_dpo_pipeline
    python3 repair_sessions.py --input outputs/sessions/all_sessions.json
    
    # Dry run (just audit, no changes):
    python3 repair_sessions.py --input outputs/sessions/all_sessions.json --dry-run

    # Use a specific model:
    python3 repair_sessions.py --input outputs/sessions/all_sessions.json --model gemini-2.5-pro
"""

import os
import sys
import json
import time
import shutil
import argparse
from typing import Dict, List

# Add parent dir to path so imports work from dsa_dpo_pipeline/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_judge import LLMJudge, localize_error


def audit_sessions(data: List[Dict]) -> Dict:
    """Audit sessions and return summary stats."""
    total = len(data)
    labels = {}
    for s in data:
        l = s.get("label", "MISSING")
        labels[l] = labels.get(l, 0) + 1

    neg = [s for s in data if s.get("label") == "negative"]
    
    missing_error_turn = [s for s in neg if "error_turn" not in s]
    invalid_error = [s for s in neg if s.get("error_turn", {}).get("index", -1) < 0]
    empty_reason = [s for s in neg if "error_turn" in s and not s["error_turn"].get("reason", "").strip()]
    valid = [s for s in neg if s.get("error_turn", {}).get("index", -1) >= 0 and s["error_turn"].get("reason", "").strip()]

    needs_repair = []
    for s in neg:
        et = s.get("error_turn", {})
        if "error_turn" not in s or et.get("index", -1) < 0 or not et.get("reason", "").strip():
            needs_repair.append(s)

    stats = {
        "total": total,
        "labels": labels,
        "negative_count": len(neg),
        "valid_error_loc": len(valid),
        "missing_error_turn": len(missing_error_turn),
        "invalid_error_index": len(invalid_error),
        "empty_error_reason": len(empty_reason),
        "needs_repair": len(needs_repair),
    }

    return stats, needs_repair


def repair_session(session: Dict, judge: LLMJudge, max_retries: int = 3, delay: float = 5.0) -> bool:
    """
    Re-run error localization for a single session.
    
    Returns True if repair succeeded, False otherwise.
    """
    dialogue = session.get("dialogue", [])
    scenario = session.get("scenario", {})
    session_id = session.get("session_id", "?")

    for attempt in range(1, max_retries + 1):
        try:
            error_info = localize_error(
                dialogue=dialogue,
                scenario=scenario,
                llm_judge=judge,
            )

            idx = error_info.get("index", -1)
            reason = error_info.get("reason", "")

            if idx >= 0 and reason.strip():
                # Validate index is an employer turn
                if idx < len(dialogue) and dialogue[idx].get("role") == "employer":
                    session["error_turn"] = error_info
                    print(f"  ✓ {session_id}: error at turn {idx} (attempt {attempt})")
                    return True
                else:
                    # Index doesn't point to an employer turn — try again
                    print(f"  ⚠ {session_id}: index {idx} is not an employer turn, retrying...")
            else:
                print(f"  ⚠ {session_id}: got index={idx}, retrying... (attempt {attempt})")

        except Exception as e:
            err_short = str(e)[:100]
            print(f"  ✗ {session_id}: attempt {attempt} failed: {err_short}")

        # Wait before retry (exponential backoff)
        if attempt < max_retries:
            wait = delay * (2 ** (attempt - 1))
            print(f"    Waiting {wait:.0f}s before retry...")
            time.sleep(wait)

    print(f"  ✗ {session_id}: all {max_retries} attempts failed")
    return False


def main():
    parser = argparse.ArgumentParser(description="Repair missing error localization in all_sessions.json")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/sessions/all_sessions.json",
        help="Path to all_sessions.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: overwrite input after backup)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to pipeline config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override judge model (e.g., 'gemini-2.5-pro')",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per session (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Base delay between retries in seconds (default: 5)",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=2.0,
        help="Delay between sessions to avoid rate limits (default: 2s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit only, don't make any changes",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading sessions from: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Audit
    print("\n=== AUDIT ===")
    stats, needs_repair = audit_sessions(data)
    print(f"Total sessions:          {stats['total']}")
    print(f"Labels:                  {stats['labels']}")
    print(f"Negative sessions:       {stats['negative_count']}")
    print(f"Valid error localization: {stats['valid_error_loc']}")
    print(f"Needs repair:            {stats['needs_repair']}")
    print(f"  - Missing error_turn:  {stats['missing_error_turn']}")
    print(f"  - Invalid index (-1):  {stats['invalid_error_index']}")
    print(f"  - Empty reason:        {stats['empty_error_reason']}")

    if not needs_repair:
        print("\n✓ All sessions are complete. Nothing to repair.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would repair {len(needs_repair)} sessions. Exiting.")
        for s in needs_repair:
            et = s.get("error_turn", {})
            reason_short = et.get("reason", "N/A")[:80]
            print(f"  {s['session_id']}: index={et.get('index', 'N/A')}, reason={reason_short}")
        return

    # Initialize judge
    print(f"\nInitializing LLM Judge...")
    judge = LLMJudge(config_path=args.config, model_override=args.model)
    print(f"Judge model: {judge.model}")

    # Repair
    print(f"\n=== REPAIRING {len(needs_repair)} SESSIONS ===\n")
    repaired = 0
    failed = 0

    for i, session in enumerate(needs_repair):
        print(f"[{i+1}/{len(needs_repair)}] Repairing {session['session_id']}...")
        success = repair_session(session, judge, max_retries=args.max_retries, delay=args.delay)
        if success:
            repaired += 1
        else:
            failed += 1

        # Delay between sessions to avoid rate limits
        if i < len(needs_repair) - 1:
            time.sleep(args.batch_delay)

    # Save
    print(f"\n=== REPAIR COMPLETE ===")
    print(f"Repaired:  {repaired}")
    print(f"Failed:    {failed}")

    output_path = args.output or args.input
    
    # Backup original
    backup_path = args.input + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(args.input, backup_path)
        print(f"Backup saved: {backup_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {output_path}")

    # Final audit
    print("\n=== POST-REPAIR AUDIT ===")
    post_stats, post_repair = audit_sessions(data)
    print(f"Valid error localization: {post_stats['valid_error_loc']}")
    print(f"Still needs repair:      {post_stats['needs_repair']}")


if __name__ == "__main__":
    main()
