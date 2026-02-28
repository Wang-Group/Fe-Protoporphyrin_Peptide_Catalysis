from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esm2_ton_generator.conditional_lora import (
    ConditionedMLMDataset,
    TonBinner,
    build_lora_model,
    conditioning_text,
)
from esm2_ton_generator.data import add_split_column, load_ton_dataset
from esm2_ton_generator.esm_backend import resolve_device


def parse_sheet_name(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional ESM-2 LoRA model with TON binning.")
    parser.add_argument("--excel-path", type=Path, required=True)
    parser.add_argument("--sheet-name", type=str, default="batch1")
    parser.add_argument("--aggregate", type=str, default="none", choices=["none", "mean", "max"])

    parser.add_argument("--model-name", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--num-bins", type=int, default=3)
    parser.add_argument("--bin-strategy", type=str, default="quantile", choices=["quantile", "uniform"])

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=None)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def json_safe_args(namespace: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def mean_loss(model, dataloader: DataLoader, device: str) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

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

    train_mask = split_df["split"] == "train"
    binner = TonBinner.fit(
        split_df.loc[train_mask, "ton"].to_numpy(),
        num_bins=args.num_bins,
        strategy=args.bin_strategy,
    )
    split_df["ton_bin"] = binner.transform(split_df["ton"].to_numpy())
    split_df["ton_bin_token"] = split_df["ton_bin"].map(binner.token_for_bin)

    tokenizer, model = build_lora_model(
        model_name=args.model_name,
        condition_tokens=binner.tokens,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    train_texts = [
        conditioning_text(seq, token)
        for seq, token in zip(
            split_df.loc[split_df["split"] == "train", "sequence"],
            split_df.loc[split_df["split"] == "train", "ton_bin_token"],
            strict=True,
        )
    ]
    val_texts = [
        conditioning_text(seq, token)
        for seq, token in zip(
            split_df.loc[split_df["split"] == "val", "sequence"],
            split_df.loc[split_df["split"] == "val", "ton_bin_token"],
            strict=True,
        )
    ]

    train_dataset = ConditionedMLMDataset(train_texts, tokenizer=tokenizer, max_length=args.max_length)
    val_dataset = ConditionedMLMDataset(val_texts, tokenizer=tokenizer, max_length=args.max_length)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss = mean_loss(model, val_loader, device=device)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(model).items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training finished without capturing a best LoRA state.")

    torch.save(best_state, args.output_dir / "lora_state.pt")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    binner.save_json(args.output_dir / "ton_binner.json")

    split_df.to_csv(args.output_dir / "dataset_splits_bins.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "conditional_lora_history.csv", index=False)

    metadata = {
        "model_name": args.model_name,
        "device_used": device,
        "best_val_loss": best_val,
        "num_bins": binner.num_bins,
        "bin_strategy": binner.strategy,
        "lora_config": {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": ["query", "key", "value"],
            "lora_dropout": args.lora_dropout,
            "bias": "none",
        },
        "training_args": json_safe_args(args),
    }
    with open(args.output_dir / "conditional_lora_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Conditional LoRA training complete.")
    print(f"Device: {device}")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Saved run directory: {args.output_dir}")


if __name__ == "__main__":
    main()
