"""Conditioned sequence generation core logic."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np


class PredictorProtocol(Protocol):
    """Minimal interface used by the generator."""

    def predict(self, sequences: Sequence[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Predict TON for sequences."""

    def residue_log_probs(self, sequence: str, residue_index: int) -> Mapping[str, float]:
        """Log-probabilities over amino acids at one residue position."""


def objective_score(
    pred_ton: float,
    lm_log_prob: float,
    target_ton: float | None,
    alpha_lm: float,
) -> float:
    """Guidance objective for selecting mutations."""
    ton_term = pred_ton if target_ton is None else -abs(pred_ton - target_ton)
    return float(ton_term + alpha_lm * lm_log_prob)


class HighTONGenerator:
    """Generate sequences by iterative masked mutation with TON guidance."""

    def __init__(self, predictor: PredictorProtocol, random_seed: int = 42) -> None:
        self.predictor = predictor
        self.rng = random.Random(random_seed)

    def _single_mutation_step(
        self,
        sequence: str,
        target_ton: float | None,
        top_k: int,
        alpha_lm: float,
    ) -> dict[str, float | int | str]:
        position = self.rng.randrange(len(sequence))
        log_prob_map = self.predictor.residue_log_probs(sequence, position)

        ranked = sorted(log_prob_map.items(), key=lambda item: item[1], reverse=True)
        aa_candidates = [aa for aa, _ in ranked[:top_k]]

        current_aa = sequence[position]
        if current_aa not in aa_candidates and current_aa in log_prob_map:
            aa_candidates.append(current_aa)

        candidates: list[str] = []
        candidate_lm: list[float] = []
        seen: set[str] = set()
        for aa in aa_candidates:
            mutated = sequence[:position] + aa + sequence[position + 1 :]
            if mutated in seen:
                continue
            seen.add(mutated)
            candidates.append(mutated)
            candidate_lm.append(float(log_prob_map[aa]))

        pred_tons = self.predictor.predict(candidates, batch_size=max(1, len(candidates)))
        objectives = [
            objective_score(float(pred), lm_log_prob, target_ton=target_ton, alpha_lm=alpha_lm)
            for pred, lm_log_prob in zip(pred_tons, candidate_lm, strict=True)
        ]
        best_idx = int(np.argmax(objectives))

        return {
            "sequence": candidates[best_idx],
            "pred_ton": float(pred_tons[best_idx]),
            "objective": float(objectives[best_idx]),
            "position": int(position),
            "lm_log_prob": float(candidate_lm[best_idx]),
        }

    def refine(
        self,
        seed_sequence: str,
        num_steps: int = 40,
        target_ton: float | None = None,
        top_k: int = 8,
        alpha_lm: float = 0.03,
        temperature: float = 0.2,
    ) -> dict[str, float | int | str]:
        current = seed_sequence
        current_pred = float(self.predictor.predict([current])[0])
        current_objective = objective_score(current_pred, lm_log_prob=0.0, target_ton=target_ton, alpha_lm=0.0)

        best_sequence = current
        best_pred = current_pred
        best_objective = current_objective

        for _ in range(num_steps):
            proposal = self._single_mutation_step(
                sequence=current,
                target_ton=target_ton,
                top_k=top_k,
                alpha_lm=alpha_lm,
            )
            proposal_obj = float(proposal["objective"])
            delta = proposal_obj - current_objective

            if delta >= 0:
                accept = True
            else:
                scaled_temp = max(temperature, 1e-6)
                accept = self.rng.random() < math.exp(delta / scaled_temp)

            if accept:
                current = str(proposal["sequence"])
                current_pred = float(proposal["pred_ton"])
                current_objective = proposal_obj

                if current_objective > best_objective:
                    best_sequence = current
                    best_pred = current_pred
                    best_objective = current_objective

        return {
            "seed_sequence": seed_sequence,
            "sequence": best_sequence,
            "pred_ton": float(best_pred),
            "objective": float(best_objective),
            "steps": int(num_steps),
        }

    def generate(
        self,
        seed_sequences: Sequence[str],
        known_sequences: set[str],
        num_sequences: int = 30,
        num_steps: int = 40,
        target_ton: float | None = None,
        top_k: int = 8,
        alpha_lm: float = 0.03,
        temperature: float = 0.2,
        min_pred_ton: float | None = None,
        required_residues: set[str] | None = None,
        max_attempts: int | None = None,
    ) -> list[dict[str, float | int | str | bool]]:
        if len(seed_sequences) == 0:
            raise ValueError("seed_sequences cannot be empty.")

        known = {seq.upper() for seq in known_sequences}
        required = {res.upper() for res in (required_residues or set())}
        seen_generated: set[str] = set()
        outputs: list[dict[str, float | int | str | bool]] = []
        attempts = 0
        max_tries = max_attempts or (num_sequences * 80)

        while len(outputs) < num_sequences and attempts < max_tries:
            attempts += 1
            seed = self.rng.choice(list(seed_sequences))
            refined = self.refine(
                seed_sequence=seed,
                num_steps=num_steps,
                target_ton=target_ton,
                top_k=top_k,
                alpha_lm=alpha_lm,
                temperature=temperature,
            )

            seq = str(refined["sequence"]).upper()
            if seq in seen_generated:
                continue

            pred_ton = float(refined["pred_ton"])
            if min_pred_ton is not None and pred_ton < min_pred_ton:
                continue
            if required and not all(residue in seq for residue in required):
                continue

            seen_generated.add(seq)
            refined["is_novel"] = seq not in known
            outputs.append(refined)

        outputs.sort(key=lambda item: float(item["pred_ton"]), reverse=True)
        return outputs
