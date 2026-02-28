"""Conditional LoRA utilities for ESM-2 + TON binning."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torch.utils.data import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .constants import AA_ALPHABET
from .esm_backend import resolve_device


def _strictly_increasing(values: np.ndarray) -> np.ndarray:
    """Collapse duplicate edge values while preserving order."""
    deduped = [float(values[0])]
    for value in values[1:]:
        if float(value) > deduped[-1]:
            deduped.append(float(value))
    return np.asarray(deduped, dtype=float)


@dataclass
class TonBinner:
    """Map TON to discrete condition bins and tokens."""

    edges: list[float]
    tokens: list[str]
    strategy: str = "quantile"

    @property
    def num_bins(self) -> int:
        return len(self.tokens)

    @classmethod
    def fit(
        cls,
        ton_values: np.ndarray,
        num_bins: int = 3,
        strategy: Literal["quantile", "uniform"] = "quantile",
        token_prefix: str = "<TON_BIN_",
    ) -> "TonBinner":
        ton_values = np.asarray(ton_values, dtype=float)
        if ton_values.ndim != 1:
            raise ValueError("ton_values must be a 1D array.")
        if len(ton_values) < 2:
            raise ValueError("Need at least 2 TON values for binning.")
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2.")

        if strategy == "quantile":
            raw_edges = np.quantile(ton_values, np.linspace(0.0, 1.0, num_bins + 1))
        elif strategy == "uniform":
            raw_edges = np.linspace(float(ton_values.min()), float(ton_values.max()), num_bins + 1)
        else:
            raise ValueError("strategy must be one of: quantile, uniform")

        edges = _strictly_increasing(raw_edges)
        if len(edges) < 3:
            raise ValueError("TON values are too concentrated to build >=2 bins.")

        tokens = [f"{token_prefix}{idx}>" for idx in range(len(edges) - 1)]
        return cls(edges=[float(x) for x in edges], tokens=tokens, strategy=strategy)

    def transform(self, ton_values: np.ndarray) -> np.ndarray:
        values = np.asarray(ton_values, dtype=float)
        boundaries = np.asarray(self.edges[1:-1], dtype=float)
        bin_idx = np.searchsorted(boundaries, values, side="right")
        return np.clip(bin_idx, 0, self.num_bins - 1).astype(int)

    def token_for_bin(self, bin_idx: int) -> str:
        if bin_idx < 0 or bin_idx >= self.num_bins:
            raise IndexError(f"Invalid bin index: {bin_idx}")
        return self.tokens[int(bin_idx)]

    def token_for_ton(self, ton_value: float) -> str:
        idx = int(self.transform(np.asarray([ton_value]))[0])
        return self.token_for_bin(idx)

    def to_dict(self) -> dict[str, object]:
        return {
            "edges": self.edges,
            "tokens": self.tokens,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TonBinner":
        return cls(
            edges=[float(x) for x in payload["edges"]],  # type: ignore[index]
            tokens=[str(x) for x in payload["tokens"]],  # type: ignore[index]
            strategy=str(payload.get("strategy", "quantile")),
        )

    def save_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "TonBinner":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


def conditioning_text(sequence: str, condition_token: str) -> str:
    return f"{condition_token} {sequence}"


def has_required_residues(sequence: str, required_residues: set[str] | None) -> bool:
    if not required_residues:
        return True
    return all(residue in sequence for residue in required_residues)


class ConditionedMLMDataset(Dataset):
    """Tokenized text examples for conditional MLM fine-tuning."""

    def __init__(self, texts: list[str], tokenizer, max_length: int | None = None) -> None:
        self.examples = []
        for text in texts:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=max_length is not None,
                max_length=max_length,
            )
            self.examples.append({"input_ids": encoded["input_ids"]})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def create_condition_tokens(num_bins: int) -> list[str]:
    return [f"<TON_BIN_{idx}>" for idx in range(num_bins)]


def _align_lm_head_bias_with_embeddings(model: torch.nn.Module) -> None:
    """
    Ensure LM-head bias matches decoder vocab size after embedding resize.

    This guards against ESM models where `resize_token_embeddings` updates
    decoder weight size but leaves `lm_head.bias` at the old vocab size.
    """
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not hasattr(lm_head, "bias"):
        return

    decoder = getattr(lm_head, "decoder", None)
    if decoder is None:
        return

    if hasattr(decoder, "weight"):
        expected_vocab = int(decoder.weight.shape[0])
    elif hasattr(decoder, "out_features"):
        expected_vocab = int(decoder.out_features)
    else:
        return
    current_bias = getattr(lm_head, "bias", None)

    if current_bias is not None and int(current_bias.shape[0]) == expected_vocab:
        if hasattr(decoder, "bias"):
            decoder.bias = current_bias
        return

    if current_bias is None:
        dtype = decoder.weight.dtype
        device = decoder.weight.device
        resized_bias = torch.nn.Parameter(torch.zeros(expected_vocab, dtype=dtype, device=device))
    else:
        resized_bias = torch.nn.Parameter(
            torch.zeros(expected_vocab, dtype=current_bias.dtype, device=current_bias.device)
        )
        copy_len = min(int(current_bias.shape[0]), expected_vocab)
        resized_bias.data[:copy_len] = current_bias.data[:copy_len]

    lm_head.bias = resized_bias
    if hasattr(decoder, "bias"):
        decoder.bias = lm_head.bias


def build_lora_model(
    model_name: str,
    condition_tokens: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: tuple[str, ...] = ("query", "key", "value"),
) -> tuple[AutoTokenizer, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    tokenizer.add_special_tokens({"additional_special_tokens": condition_tokens})
    model.resize_token_embeddings(len(tokenizer))
    _align_lm_head_bias_with_embeddings(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
    )
    lora_model = get_peft_model(model, config)
    return tokenizer, lora_model


def amino_acid_token_ids(tokenizer) -> list[int]:
    token_ids: list[int] = []
    for aa in AA_ALPHABET:
        idx = int(tokenizer.convert_tokens_to_ids(aa))
        if idx == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer does not support amino acid token: {aa}")
        token_ids.append(idx)
    return token_ids


def load_lora_artifacts(
    run_dir: str | Path,
    device: str = "auto",
) -> tuple[AutoTokenizer, torch.nn.Module, TonBinner, dict[str, object]]:
    run_path = Path(run_dir)
    metadata_path = run_path / "conditional_lora_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    tokenizer = AutoTokenizer.from_pretrained(run_path / "tokenizer")
    model = AutoModelForMaskedLM.from_pretrained(str(metadata["model_name"]))
    model.resize_token_embeddings(len(tokenizer))
    _align_lm_head_bias_with_embeddings(model)
    model = get_peft_model(model, LoraConfig(**metadata["lora_config"]))
    adapter_state = torch.load(run_path / "lora_state.pt", map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, adapter_state)
    model.to(resolve_device(device))
    model.eval()

    binner = TonBinner.load_json(run_path / "ton_binner.json")
    return tokenizer, model, binner, metadata


def _sample_from_logits(
    logits: torch.Tensor,
    top_k: int,
    temperature: float,
    rng: random.Random,
) -> tuple[int, float]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    scaled = logits / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=-1)

    top_k = min(top_k, probs.numel())
    top_values, top_indices = torch.topk(probs, k=top_k)
    top_values = top_values / top_values.sum()

    cumulative = 0.0
    threshold = rng.random()
    chosen_rank = 0
    for idx, value in enumerate(top_values.tolist()):
        cumulative += float(value)
        if threshold <= cumulative:
            chosen_rank = idx
            break

    chosen_idx = int(top_indices[chosen_rank].item())
    chosen_prob = float(top_values[chosen_rank].item())
    return chosen_idx, math.log(max(chosen_prob, 1e-12))


@torch.no_grad()
def sample_sequence_from_condition(
    model,
    tokenizer,
    condition_token: str,
    sequence_length: int = 10,
    top_k: int = 8,
    temperature: float = 1.0,
    refine_steps: int = 20,
    random_seed: int | None = None,
) -> tuple[str, float]:
    rng = random.Random(random_seed)
    device = next(model.parameters()).device

    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    cond_id = tokenizer.convert_tokens_to_ids(condition_token)
    if cond_id == tokenizer.unk_token_id:
        raise ValueError(f"Unknown condition token: {condition_token}")

    aa_ids = amino_acid_token_ids(tokenizer)
    aa_tensor = torch.tensor(aa_ids, dtype=torch.long, device=device)

    input_ids = torch.tensor(
        [[cls_id, cond_id] + [mask_id] * sequence_length + [eos_id]],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)

    residue_positions = list(range(2, 2 + sequence_length))
    log_prob_acc = 0.0

    for pos in residue_positions:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, pos, aa_tensor]
        selected_rank, log_prob = _sample_from_logits(logits, top_k=top_k, temperature=temperature, rng=rng)
        selected_token_id = int(aa_tensor[selected_rank].item())
        input_ids[0, pos] = selected_token_id
        log_prob_acc += log_prob

    for _ in range(refine_steps):
        pos = rng.choice(residue_positions)
        original_id = int(input_ids[0, pos].item())
        input_ids[0, pos] = mask_id

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, pos, aa_tensor]
        selected_rank, log_prob = _sample_from_logits(logits, top_k=top_k, temperature=temperature, rng=rng)
        selected_token_id = int(aa_tensor[selected_rank].item())
        input_ids[0, pos] = selected_token_id
        if selected_token_id != original_id:
            log_prob_acc += log_prob

    token_ids = input_ids[0, residue_positions].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    sequence = "".join(tokens)
    avg_log_prob = float(log_prob_acc / max(sequence_length, 1))
    return sequence, avg_log_prob


@torch.no_grad()
def pseudo_log_likelihood(
    model,
    tokenizer,
    condition_token: str,
    sequence: str,
) -> float:
    device = next(model.parameters()).device
    mask_id = tokenizer.mask_token_id
    cond_id = tokenizer.convert_tokens_to_ids(condition_token)
    if cond_id == tokenizer.unk_token_id:
        raise ValueError(f"Unknown condition token: {condition_token}")

    aa_ids = amino_acid_token_ids(tokenizer)
    aa_set = set(aa_ids)

    encoded = tokenizer(conditioning_text(sequence, condition_token), return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    residue_positions = [i for i, token in enumerate(input_ids[0].tolist()) if token in aa_set]
    if len(residue_positions) == 0:
        raise ValueError("No residue tokens found for sequence.")

    total = 0.0
    for pos in residue_positions:
        original_id = int(input_ids[0, pos].item())
        masked = input_ids.clone()
        masked[0, pos] = mask_id
        logits = model(input_ids=masked, attention_mask=attention_mask).logits[0, pos]
        log_prob = torch.log_softmax(logits, dim=-1)[original_id].item()
        total += float(log_prob)
    return total / len(residue_positions)
