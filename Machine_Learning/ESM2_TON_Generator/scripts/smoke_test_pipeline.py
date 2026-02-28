from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esm2_ton_generator.data import add_split_column, high_ton_threshold, load_ton_dataset
from esm2_ton_generator.esm_backend import ESM2FeatureExtractor, resolve_device
from esm2_ton_generator.generation_core import HighTONGenerator
from esm2_ton_generator.predictor import ESM2TONPredictor
from esm2_ton_generator.train import fit_regressor, set_random_seed


def parse_sheet_name(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick smoke test: train + generate.")
    parser.add_argument("--excel-path", type=Path, required=True)
    parser.add_argument("--sheet-name", type=str, default="batch1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    device = resolve_device(args.device)
    data_df = load_ton_dataset(args.excel_path, sheet_name=parse_sheet_name(args.sheet_name), aggregate="mean")
    data_df = data_df.head(args.max_samples).reset_index(drop=True)
    split_df = add_split_column(data_df, random_state=args.seed)

    extractor = ESM2FeatureExtractor(args.model_name, device=device)
    embeddings = extractor.embed(split_df["sequence"].tolist(), batch_size=16, show_progress=True)

    train_idx = split_df.index[split_df["split"] == "train"].to_numpy()
    val_idx = split_df.index[split_df["split"] == "val"].to_numpy()

    model, history, target_mean, target_std = fit_regressor(
        train_embeddings=embeddings[train_idx],
        train_targets=split_df.iloc[train_idx]["ton"].to_numpy(),
        val_embeddings=embeddings[val_idx],
        val_targets=split_df.iloc[val_idx]["ton"].to_numpy(),
        hidden_dim=128,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        epochs=6,
        patience=3,
        device=device,
    )

    checkpoint_path = args.output_dir / "smoke_regressor.pt"
    checkpoint = {
        "model_name": args.model_name,
        "input_dim": int(embeddings.shape[1]),
        "hidden_dim": 128,
        "dropout": 0.1,
        "regressor_state": model.state_dict(),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
    }
    torch.save(checkpoint, checkpoint_path)

    predictor = ESM2TONPredictor.from_checkpoint(checkpoint_path, device=device)
    generator = HighTONGenerator(predictor=predictor, random_seed=args.seed)
    target_ton = high_ton_threshold(split_df, quantile=0.85)
    seeds = (
        split_df.sort_values("ton", ascending=False)["sequence"]
        .drop_duplicates()
        .head(12)
        .tolist()
    )

    generated = generator.generate(
        seed_sequences=seeds,
        known_sequences=set(split_df["sequence"].tolist()),
        num_sequences=6,
        num_steps=12,
        target_ton=target_ton,
        top_k=6,
        alpha_lm=0.03,
        temperature=0.2,
    )
    generated_df = pd.DataFrame(generated)
    generated_path = args.output_dir / "smoke_generated.csv"
    generated_df.to_csv(generated_path, index=False)

    with open(args.output_dir / "smoke_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print("Smoke test complete.")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Generated: {generated_path}")
    if len(generated_df) > 0:
        print(generated_df.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
