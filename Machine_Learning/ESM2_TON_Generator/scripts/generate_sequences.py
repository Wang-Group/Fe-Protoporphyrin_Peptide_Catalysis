from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esm2_ton_generator.data import high_ton_threshold, load_ton_dataset
from esm2_ton_generator.generation_core import HighTONGenerator
from esm2_ton_generator.predictor import ESM2TONPredictor


def parse_sheet_name(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-TON peptide candidates.")
    parser.add_argument("--excel-path", type=Path, required=True, help="Path to data_all.xlsx")
    parser.add_argument("--sheet-name", type=str, default="batch1")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["none", "mean", "max"])

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=None, help="Override model name in checkpoint")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--num-sequences", type=int, default=30)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--alpha-lm", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed-pool-size", type=int, default=20)

    parser.add_argument("--target-ton", type=float, default=None)
    parser.add_argument("--high-ton-quantile", type=float, default=0.9)
    parser.add_argument("--min-pred-ton", type=float, default=None)
    parser.add_argument(
        "--require-aa",
        action="append",
        default=[],
        help="Amino acid that must appear in each generated sequence. Repeatable, e.g. --require-aa K --require-aa W",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def parse_required_residues(values: list[str]) -> set[str]:
    parsed: set[str] = set()
    for value in values:
        token = value.strip().upper()
        if len(token) != 1 or not token.isalpha():
            raise ValueError(f"Invalid residue in --require-aa: {value!r}")
        parsed.add(token)
    return parsed


def main() -> None:
    args = parse_args()
    required_residues = parse_required_residues(args.require_aa)
    data_df = load_ton_dataset(
        excel_path=args.excel_path,
        sheet_name=parse_sheet_name(args.sheet_name),
        aggregate=args.aggregate,
    )

    target_ton = args.target_ton
    if target_ton is None:
        target_ton = high_ton_threshold(data_df, quantile=args.high_ton_quantile)

    seed_df = data_df[data_df["ton"] >= target_ton].sort_values("ton", ascending=False)
    if len(seed_df) == 0:
        seed_df = data_df.sort_values("ton", ascending=False)

    seed_sequences = seed_df["sequence"].drop_duplicates().head(args.seed_pool_size).tolist()
    known_sequences = set(data_df["sequence"].tolist())

    predictor = ESM2TONPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device,
    )
    generator = HighTONGenerator(predictor=predictor, random_seed=args.seed)

    results = generator.generate(
        seed_sequences=seed_sequences,
        known_sequences=known_sequences,
        num_sequences=args.num_sequences,
        num_steps=args.num_steps,
        target_ton=target_ton,
        top_k=args.top_k,
        alpha_lm=args.alpha_lm,
        temperature=args.temperature,
        min_pred_ton=args.min_pred_ton,
        required_residues=required_residues,
    )

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort_values("pred_ton", ascending=False).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_csv, index=False)

    print(f"Target TON for conditioning: {target_ton:.4f}")
    print(f"Seed pool size: {len(seed_sequences)}")
    print(f"Required residues: {sorted(required_residues) if required_residues else 'None'}")
    print(f"Generated sequences: {len(result_df)}")
    print(f"Saved to: {args.output_csv}")
    if len(result_df) > 0:
        print(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
