#!/usr/bin/env python3
"""Compute distribution of segment lengths from phase2 segment JSON files.

By default, this script reads files under:
  ./phase2_output/segments/*_segment.json

Example:
  python segment_length_distribution.py
  python segment_length_distribution.py --segments-dir ./phase2_output/segments --side both
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get distribution of segment lengths from DSA-DPO segment files."
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "phase2_output" / "segments",
        help="Directory containing *_segment.json files.",
    )
    parser.add_argument(
        "--side",
        choices=["positive", "negative", "both"],
        default="positive",
        help=(
            "Which segment to measure length from. 'both' validates that positive and "
            "negative lengths match and then counts that shared length."
        ),
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print output as JSON instead of plain text.",
    )
    return parser.parse_args()


def find_segment_files(segments_dir: Path) -> list[Path]:
    if not segments_dir.exists():
        raise FileNotFoundError(f"Directory not found: {segments_dir}")
    files = sorted(segments_dir.glob("*_segment.json"))
    if not files:
        raise FileNotFoundError(
            f"No *_segment.json files found in directory: {segments_dir}"
        )
    return files


def iter_lengths(files: Iterable[Path], side: str) -> tuple[list[int], list[dict[str, object]]]:
    lengths: list[int] = []
    issues: list[dict[str, object]] = []

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        pos = data.get("positive_segment")
        neg = data.get("negative_segment")

        if not isinstance(pos, list) or not isinstance(neg, list):
            issues.append(
                {
                    "file": str(file_path),
                    "reason": "Missing or malformed positive_segment/negative_segment",
                }
            )
            continue

        if side == "positive":
            lengths.append(len(pos))
            continue

        if side == "negative":
            lengths.append(len(neg))
            continue

        # side == "both"
        if len(pos) != len(neg):
            issues.append(
                {
                    "file": str(file_path),
                    "reason": "Length mismatch between positive and negative segments",
                    "positive_length": len(pos),
                    "negative_length": len(neg),
                }
            )
            continue

        lengths.append(len(pos))

    return lengths, issues


def build_report(lengths: list[int], total_files: int, issues: list[dict[str, object]]) -> dict[str, object]:
    if not lengths:
        return {
            "total_files": total_files,
            "valid_files": 0,
            "invalid_files": len(issues),
            "length_distribution": {},
            "summary": {},
            "issues": issues,
        }

    dist = Counter(lengths)
    ordered_dist = dict(sorted(dist.items(), key=lambda kv: kv[0]))

    report: dict[str, object] = {
        "total_files": total_files,
        "valid_files": len(lengths),
        "invalid_files": len(issues),
        "length_distribution": ordered_dist,
        "summary": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": mean(lengths),
            "median": median(lengths),
        },
    }

    if issues:
        report["issues"] = issues

    return report


def print_text_report(report: dict[str, object]) -> None:
    print("Segment Length Distribution")
    print("=" * 28)
    print(f"Total files:   {report['total_files']}")
    print(f"Valid files:   {report['valid_files']}")
    print(f"Invalid files: {report['invalid_files']}")

    distribution = report.get("length_distribution", {})
    print("\nLength counts:")
    if distribution:
        for seg_len, count in distribution.items():
            print(f"  length={seg_len}: {count}")
    else:
        print("  (none)")

    summary = report.get("summary", {})
    if summary:
        print("\nSummary:")
        print(f"  min:    {summary['min']}")
        print(f"  max:    {summary['max']}")
        print(f"  mean:   {summary['mean']:.4f}")
        print(f"  median: {summary['median']}")

    issues = report.get("issues", [])
    if issues:
        print(f"\nIssues ({len(issues)}):")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")


def main() -> None:
    args = parse_args()
    files = find_segment_files(args.segments_dir)
    lengths, issues = iter_lengths(files, args.side)
    report = build_report(lengths=lengths, total_files=len(files), issues=issues)

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
