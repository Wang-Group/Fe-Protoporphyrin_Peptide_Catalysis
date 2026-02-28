"""ESM-2 backend utilities."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .constants import AA_ALPHABET


def resolve_device(device: str = "auto") -> str:
    """Resolve device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class ESM2FeatureExtractor:
    """Embeddings and masked-token log-probabilities from ESM-2."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        aa_to_token_id: dict[str, int] = {}
        for aa in AA_ALPHABET:
            token_id = int(self.tokenizer.convert_tokens_to_ids(aa))
            if token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Tokenizer does not support amino acid token: {aa}")
            aa_to_token_id[aa] = token_id
        self.aa_to_token_id = aa_to_token_id
        self.aa_token_ids = torch.tensor(
            list(aa_to_token_id.values()),
            dtype=torch.long,
            device=self.device,
        )

        if self.tokenizer.mask_token_id is None:
            raise ValueError(f"Tokenizer for {model_name} does not expose a mask token ID.")

    def _pool_residue_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        residue_mask = torch.isin(input_ids, self.aa_token_ids).to(hidden_states.dtype)
        weighted_sum = (hidden_states * residue_mask.unsqueeze(-1)).sum(dim=1)
        counts = residue_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return weighted_sum / counts

    @torch.no_grad()
    def embed(
        self,
        sequences: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Return pooled sequence embeddings on CPU."""
        embeddings: list[torch.Tensor] = []
        iterator = range(0, len(sequences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding sequences", leave=False)

        for start in iterator:
            batch = list(sequences[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            outputs = self.model(**encoded, output_hidden_states=True)
            pooled = self._pool_residue_embeddings(outputs.hidden_states[-1], encoded["input_ids"])
            embeddings.append(pooled.cpu())

        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def masked_token_log_probs(self, sequence: str, residue_index: int) -> dict[str, float]:
        """Return amino-acid log-probs at one residue position."""
        encoded = self.tokenizer(sequence, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        residue_positions = torch.where(torch.isin(input_ids[0], self.aa_token_ids))[0]

        if residue_index < 0 or residue_index >= len(residue_positions):
            raise IndexError(
                f"residue_index={residue_index} out of bounds for sequence length {len(residue_positions)}"
            )

        token_position = int(residue_positions[residue_index].item())
        masked_ids = input_ids.clone()
        masked_ids[0, token_position] = int(self.tokenizer.mask_token_id)

        outputs = self.model(
            input_ids=masked_ids,
            attention_mask=encoded["attention_mask"],
        )
        logits = outputs.logits[0, token_position]
        log_probs = torch.log_softmax(logits, dim=-1)
        return {aa: float(log_probs[token_id].item()) for aa, token_id in self.aa_to_token_id.items()}
