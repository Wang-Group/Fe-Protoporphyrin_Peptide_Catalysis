from __future__ import annotations

import numpy as np

from esm2_ton_generator.constants import AA_ALPHABET
from esm2_ton_generator.generation_core import HighTONGenerator, objective_score


class FakePredictor:
    def predict(self, sequences, batch_size: int = 32, show_progress: bool = False):
        return np.asarray([seq.count("W") * 3.0 + seq.count("Y") * 2.0 for seq in sequences], dtype=float)

    def residue_log_probs(self, sequence: str, residue_index: int):
        scores = {aa: -4.0 for aa in AA_ALPHABET}
        scores["W"] = -0.01
        scores["Y"] = -0.10
        scores[sequence[residue_index]] = -0.20
        return scores


def test_objective_score_prefers_closer_target() -> None:
    near = objective_score(pred_ton=22.0, lm_log_prob=-0.5, target_ton=20.0, alpha_lm=0.1)
    far = objective_score(pred_ton=10.0, lm_log_prob=-0.5, target_ton=20.0, alpha_lm=0.1)
    assert near > far


def test_refine_increases_fake_ton() -> None:
    generator = HighTONGenerator(predictor=FakePredictor(), random_seed=1)
    result = generator.refine(
        seed_sequence="AAAAAAAAAA",
        num_steps=8,
        target_ton=None,
        top_k=2,
        alpha_lm=0.05,
        temperature=0.05,
    )
    assert len(str(result["sequence"])) == 10
    assert float(result["pred_ton"]) > 0.0


def test_generate_returns_unique_sequences() -> None:
    generator = HighTONGenerator(predictor=FakePredictor(), random_seed=3)
    outputs = generator.generate(
        seed_sequences=["AAAAAAAAAA", "CCCCCCCCCC"],
        known_sequences={"AAAAAAAAAA"},
        num_sequences=4,
        num_steps=6,
        target_ton=None,
        top_k=2,
        alpha_lm=0.05,
        temperature=0.1,
    )
    sequences = [str(item["sequence"]) for item in outputs]
    assert len(sequences) == len(set(sequences))


class KFavoringPredictor:
    def predict(self, sequences, batch_size: int = 32, show_progress: bool = False):
        return np.asarray([seq.count("K") * 5.0 + 1.0 for seq in sequences], dtype=float)

    def residue_log_probs(self, sequence: str, residue_index: int):
        scores = {aa: -3.0 for aa in AA_ALPHABET}
        scores["K"] = -0.01
        scores[sequence[residue_index]] = -0.1
        return scores


def test_generate_can_enforce_required_residue() -> None:
    generator = HighTONGenerator(predictor=KFavoringPredictor(), random_seed=11)
    outputs = generator.generate(
        seed_sequences=["AAAAAAAAAA"],
        known_sequences=set(),
        num_sequences=5,
        num_steps=8,
        required_residues={"K"},
        top_k=3,
    )
    assert len(outputs) == 5
    assert all("K" in str(item["sequence"]) for item in outputs)
