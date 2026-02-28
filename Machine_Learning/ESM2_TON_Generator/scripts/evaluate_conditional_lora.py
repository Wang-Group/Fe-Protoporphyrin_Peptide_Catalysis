from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esm2_ton_generator.conditional_lora import load_lora_artifacts, pseudo_log_likelihood


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conditional LoRA model by pseudo-log-likelihood.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=None, help="Defaults to run-dir/dataset_splits_bins.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_csv = args.split_csv or (args.run_dir / "dataset_splits_bins.csv")
    split_df = pd.read_csv(split_csv)
    required_columns = {"sequence", "split", "ton_bin_token"}
    missing = required_columns - set(split_df.columns)
    if missing:
        raise ValueError(f"Missing columns in split file: {sorted(missing)}")

    tokenizer, model, _, _ = load_lora_artifacts(run_dir=args.run_dir, device=args.device)

    pll_values = []
    for row in split_df.itertuples(index=False):
        pll = pseudo_log_likelihood(
            model=model,
            tokenizer=tokenizer,
            condition_token=str(row.ton_bin_token),
            sequence=str(row.sequence),
        )
        ppl = math.exp(-pll)
        pll_values.append((float(pll), float(ppl)))

    split_df = split_df.copy()
    split_df["pseudo_log_likelihood"] = [item[0] for item in pll_values]
    split_df["pseudo_perplexity"] = [item[1] for item in pll_values]

    summary: dict[str, dict[str, float]] = {}
    for split_name in ["train", "val", "test"]:
        subset = split_df[split_df["split"] == split_name]
        if len(subset) == 0:
            continue
        summary[split_name] = {
            "mean_pll": float(np.mean(subset["pseudo_log_likelihood"])),
            "std_pll": float(np.std(subset["pseudo_log_likelihood"])),
            "mean_pseudo_perplexity": float(np.mean(subset["pseudo_perplexity"])),
        }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output_csv, index=False)
    with open(args.output_csv.with_suffix(".summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved detailed evaluation to: {args.output_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
