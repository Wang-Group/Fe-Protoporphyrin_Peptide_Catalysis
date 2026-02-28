from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esm2_ton_generator.data import load_ton_dataset
from esm2_ton_generator.metrics import regression_metrics
from esm2_ton_generator.predictor import ESM2TONPredictor


def parse_sheet_name(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TON regressor checkpoint.")
    parser.add_argument("--excel-path", type=Path, required=True, help="Path to data_all.xlsx")
    parser.add_argument("--sheet-name", type=str, default="batch1")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["none", "mean", "max"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=None, help="Override model name in checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-csv", type=Path, default=None, help="Optional split file from training run")
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_df = load_ton_dataset(
        excel_path=args.excel_path,
        sheet_name=parse_sheet_name(args.sheet_name),
        aggregate=args.aggregate,
    )

    predictor = ESM2TONPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device,
    )
    pred = predictor.predict(
        data_df["sequence"].tolist(),
        batch_size=args.batch_size,
        show_progress=True,
    )

    output_df = data_df.copy()
    output_df["pred_ton"] = pred

    metrics = {"overall": regression_metrics(output_df["ton"].to_numpy(), output_df["pred_ton"].to_numpy())}

    if args.split_csv is not None:
        split_df = pd.read_csv(args.split_csv)
        split_cols = split_df[["sequence", "split"]].drop_duplicates()
        output_df = output_df.merge(split_cols, on="sequence", how="left")

        for split_name in ["train", "val", "test"]:
            subset = output_df[output_df["split"] == split_name]
            if len(subset) == 0:
                continue
            metrics[split_name] = regression_metrics(
                subset["ton"].to_numpy(),
                subset["pred_ton"].to_numpy(),
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
