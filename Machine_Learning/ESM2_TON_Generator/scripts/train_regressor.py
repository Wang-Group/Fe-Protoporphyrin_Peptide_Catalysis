from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esm2_ton_generator.data import add_split_column, load_ton_dataset
from esm2_ton_generator.esm_backend import ESM2FeatureExtractor, resolve_device
from esm2_ton_generator.metrics import regression_metrics
from esm2_ton_generator.train import fit_regressor, predict_with_regressor, set_random_seed


def parse_sheet_name(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TON regressor with ESM-2 embeddings.")
    parser.add_argument("--excel-path", type=Path, required=True, help="Path to data_all.xlsx")
    parser.add_argument("--sheet-name", type=str, default="batch1", help="Excel sheet name or index")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["none", "mean", "max"])

    parser.add_argument("--model-name", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--embed-batch-size", type=int, default=32)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    device = resolve_device(args.device)

    data_df = load_ton_dataset(
        excel_path=args.excel_path,
        sheet_name=parse_sheet_name(args.sheet_name),
        aggregate=args.aggregate,
    )
    split_df = add_split_column(
        data_df,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        random_state=args.seed,
    )

    extractor = ESM2FeatureExtractor(model_name=args.model_name, device=device)
    embeddings = extractor.embed(
        split_df["sequence"].tolist(),
        batch_size=args.embed_batch_size,
        show_progress=True,
    )

    train_idx = np.where(split_df["split"].values == "train")[0]
    val_idx = np.where(split_df["split"].values == "val")[0]
    test_idx = np.where(split_df["split"].values == "test")[0]

    model, history, target_mean, target_std = fit_regressor(
        train_embeddings=embeddings[train_idx],
        train_targets=split_df.iloc[train_idx]["ton"].to_numpy(),
        val_embeddings=embeddings[val_idx],
        val_targets=split_df.iloc[val_idx]["ton"].to_numpy(),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
    )

    metrics: dict[str, dict[str, float]] = {}
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        y_true = split_df.iloc[idx]["ton"].to_numpy()
        y_pred = predict_with_regressor(
            model=model,
            embeddings=embeddings[idx],
            target_mean=target_mean,
            target_std=target_std,
            device=device,
            batch_size=args.batch_size,
        )
        metrics[split_name] = regression_metrics(y_true=y_true, y_pred=y_pred)

    checkpoint = {
        "model_name": args.model_name,
        "input_dim": int(embeddings.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "regressor_state": model.state_dict(),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "training_args": vars(args),
        "metrics": metrics,
    }
    torch.save(checkpoint, args.output_dir / "ton_regressor.pt")

    split_df.to_csv(args.output_dir / "dataset_splits.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Training complete.")
    print(f"Device: {device}")
    print(f"Saved checkpoint: {args.output_dir / 'ton_regressor.pt'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
