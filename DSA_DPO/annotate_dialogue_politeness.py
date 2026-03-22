"""
Annotate dialogue datasets with per-turn politeness scores and labels.

This script reads a JSON dataset (list of dialogue records), scores each turn
using ConvoKit via PolitenessScorer, and writes an enriched JSON output.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from politeness_scorer import PolitenessScorer, get_politeness_label


def _extract_turn_text(turn: Dict[str, Any]) -> str:
    """Extract turn text from common field names."""
    for key in ("response", "content", "text", "utterance"):
        value = turn.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_turn_strategy(turn: Dict[str, Any]) -> Optional[str]:
    """Extract optional strategy hint for fallback/tiebreak scoring."""
    for key in ("persuasion_strategy", "negotiation_strategy", "strategy"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def annotate_turns(
    dialogue: List[Dict[str, Any]],
    scorer: PolitenessScorer,
    round_digits: int = 4,
) -> Dict[str, Any]:
    """
    Annotate each turn with politeness score and label.

    Returns dialogue-level politeness summary statistics.
    """
    turn_scores: List[float] = []
    role_sums: Dict[str, float] = {}
    role_counts: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}

    for turn in dialogue:
        text = _extract_turn_text(turn)
        strategy = _extract_turn_strategy(turn)
        role = str(turn.get("role", "unknown")).lower()

        result = scorer.score_utterance(text=text, strategy=strategy)
        score = float(result.score)
        label = get_politeness_label(score)

        turn["politeness_score"] = round(score, round_digits)
        turn["politeness_label"] = label
        

        turn_scores.append(score)
        role_sums[role] = role_sums.get(role, 0.0) + score
        role_counts[role] = role_counts.get(role, 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1

    if not turn_scores:
        return {
            "num_turns": 0,
            "avg_politeness_score": 0.0,
            "avg_politeness_label": get_politeness_label(0.0),
            "min_politeness_score": 0.0,
            "max_politeness_score": 0.0,
            "label_counts": {},
            "role_avg_politeness": {},
        }

    avg_score = sum(turn_scores) / len(turn_scores)
    role_avg = {
        role: round(role_sums[role] / role_counts[role], round_digits)
        for role in role_sums
    }

    return {
        "num_turns": len(turn_scores),
        "avg_politeness_score": round(avg_score, round_digits),
        "avg_politeness_label": get_politeness_label(avg_score),
        "min_politeness_score": round(min(turn_scores), round_digits),
        "max_politeness_score": round(max(turn_scores), round_digits),
        "label_counts": label_counts,
        "role_avg_politeness": role_avg,
    }


def annotate_dataset(
    data: List[Dict[str, Any]],
    scorer: PolitenessScorer,
    dialogue_key: str = "conversation",
    round_digits: int = 4,
    log_every: int = 200,
) -> Dict[str, int]:
    """Annotate all records in dataset in-place and return processing stats."""
    total = len(data)
    processed = 0
    skipped = 0

    for idx, record in enumerate(data):
        dialogue = record.get(dialogue_key)
        if not isinstance(dialogue, list):
            skipped += 1
            continue

        summary = annotate_turns(
            dialogue=dialogue,
            scorer=scorer,
            round_digits=round_digits,
        )
        record["conversation_politeness_summary"] = summary
        processed += 1

        if (idx + 1) % log_every == 0:
            print(f"Processed {idx + 1}/{total} records...", flush=True)

    return {"total": total, "processed": processed, "skipped": skipped}


def build_output_path(input_path: Path) -> Path:
    """Create default output path next to input file."""
    return input_path.with_name(f"{input_path.stem}_with_politeness{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate each dialogue turn with politeness score and politeness label "
            "using ConvoKit."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output JSON file (default: <input>_with_politeness.json)",
    )
    parser.add_argument(
        "--dialogue-key",
        default="conversation",
        help="Key containing turn list in each record (default: conversation)",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=4,
        help="Decimal places for saved politeness scores (default: 4)",
    )
    parser.add_argument(
        "--no-convokit",
        action="store_true",
        help="Disable ConvoKit and force fallback politeness estimation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else build_output_path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading dataset: {input_path}", flush=True)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array of dialogue records.")

    print("Initializing politeness scorer...", flush=True)
    scorer = PolitenessScorer(use_convokit=not args.no_convokit)
    print(f"ConvoKit enabled: {scorer.use_convokit}", flush=True)

    stats = annotate_dataset(
        data=data,
        scorer=scorer,
        dialogue_key=args.dialogue_key,
        round_digits=args.round_digits,
    )

    print(f"Writing enriched dataset: {output_path}", flush=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        "Done. "
        f"Total records: {stats['total']}, "
        f"Processed: {stats['processed']}, "
        f"Skipped: {stats['skipped']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
