from __future__ import annotations

import numpy as np
import torch

from esm2_ton_generator.conditional_lora import (
    TonBinner,
    _align_lm_head_bias_with_embeddings,
    conditioning_text,
    has_required_residues,
)


def test_ton_binner_quantile_fit_and_transform() -> None:
    ton = np.array([1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 15.0, 18.0, 20.0])
    binner = TonBinner.fit(ton, num_bins=3, strategy="quantile")

    bins = binner.transform(np.array([1.0, 5.0, 20.0]))
    assert binner.num_bins >= 2
    assert bins.min() >= 0
    assert bins.max() < binner.num_bins


def test_ton_binner_roundtrip_json(tmp_path) -> None:
    ton = np.array([1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 15.0, 18.0, 20.0])
    binner = TonBinner.fit(ton, num_bins=3, strategy="uniform")
    path = tmp_path / "binner.json"
    binner.save_json(path)

    loaded = TonBinner.load_json(path)
    assert loaded.tokens == binner.tokens
    assert loaded.edges == binner.edges
    assert loaded.strategy == binner.strategy


def test_conditioning_text_and_required_residue_check() -> None:
    text = conditioning_text("ACDEFGHIKL", "<TON_BIN_2>")
    assert text == "<TON_BIN_2> ACDEFGHIKL"

    assert has_required_residues("ACDEFGHIKL", {"K"})
    assert not has_required_residues("ACDEFGHILL", {"K"})


def test_align_lm_head_bias_with_embeddings_resizes_bias() -> None:
    class DummyLMHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = torch.nn.Linear(4, 6, bias=False)
            self.bias = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = DummyLMHead()

    model = DummyModel()
    _align_lm_head_bias_with_embeddings(model)

    assert model.lm_head.bias.shape[0] == 6
    assert torch.allclose(model.lm_head.bias[:3], torch.tensor([1.0, 2.0, 3.0]))
    assert model.lm_head.decoder.bias is model.lm_head.bias


def test_align_lm_head_bias_uses_decoder_weight_shape_when_out_features_stale() -> None:
    class DummyLMHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = torch.nn.Linear(4, 4, bias=False)
            # Mimic HF resize behavior where weight grows but out_features is stale.
            self.decoder.weight = torch.nn.Parameter(torch.randn(6, 4))
            self.bias = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = DummyLMHead()

    model = DummyModel()
    _align_lm_head_bias_with_embeddings(model)
    assert model.lm_head.bias.shape[0] == 6
