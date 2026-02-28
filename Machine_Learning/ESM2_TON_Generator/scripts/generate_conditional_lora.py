from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esm2_ton_generator.conditional_lora import (
    has_required_residues,
    load_lora_artifacts,
    sample_sequence_from_condition,
)
from esm2_ton_generator.data import load_ton_dataset


def parse_sheet_name(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def parse_required_residues(values: list[str]) -> set[str]:
    parsed: set[str] = set()
    for value in values:
        token = value.strip().upper()
        if len(token) != 1 or not token.isalpha():
            raise ValueError(f"Invalid residue in --require-aa: {value!r}")
        parsed.add(token)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate peptides from conditional LoRA model.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Output directory from train_conditional_lora.py")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--refine-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target-ton", type=float, default=None)
    parser.add_argument("--ton-bin", type=int, default=None)
    parser.add_argument("--require-aa", action="append", default=[])
    parser.add_argument("--max-attempts", type=int, default=2000)

    parser.add_argument("--excel-path", type=Path, default=None, help="Optional; used to annotate novelty")
    parser.add_argument("--sheet-name", type=str, default="batch1")
    parser.add_argument("--aggregate", type=str, default="none", choices=["none", "mean", "max"])

    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_residues = parse_required_residues(args.require_aa)

    tokenizer, model, binner, _ = load_lora_artifacts(run_dir=args.run_dir, device=args.device)
    if args.ton_bin is not None:
        ton_bin = int(args.ton_bin)
    elif args.target_ton is not None:
        ton_bin = int(binner.transform([args.target_ton])[0])
    else:
        ton_bin = binner.num_bins - 1
    condition_token = binner.token_for_bin(ton_bin)

    known_sequences: set[str] = set()
    if args.excel_path is not None:
        data_df = load_ton_dataset(
            excel_path=args.excel_path,
            sheet_name=parse_sheet_name(args.sheet_name),
            aggregate=args.aggregate,
        )
        known_sequences = set(data_df["sequence"].str.upper().tolist())

    outputs: list[dict[str, object]] = []
    seen: set[str] = set()
    attempts = 0

    while len(outputs) < args.num_sequences and attempts < args.max_attempts:
        attempts += 1
        sequence, avg_log_prob = sample_sequence_from_condition(
            model=model,
            tokenizer=tokenizer,
            condition_token=condition_token,
            sequence_length=args.sequence_length,
            top_k=args.top_k,
            temperature=args.temperature,
            refine_steps=args.refine_steps,
            random_seed=args.seed + attempts,
        )
        sequence = sequence.upper()
        if sequence in seen:
            continue
        if not has_required_residues(sequence, required_residues):
            continue

        seen.add(sequence)
        outputs.append(
            {
                "sequence": sequence,
                "condition_token": condition_token,
                "ton_bin": ton_bin,
                "avg_log_prob": float(avg_log_prob),
                "contains_required_residues": True,
                "is_novel": sequence not in known_sequences if known_sequences else None,
            }
        )

    output_df = pd.DataFrame(outputs).sort_values("avg_log_prob", ascending=False).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    print(f"Condition token: {condition_token} (bin={ton_bin})")
    print(f"Required residues: {sorted(required_residues) if required_residues else 'None'}")
    print(f"Generated sequences: {len(output_df)} / {args.num_sequences}")
    print(f"Saved: {args.output_csv}")
    if len(output_df) > 0:
        print(output_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
