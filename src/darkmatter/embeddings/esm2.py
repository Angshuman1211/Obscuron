"""ESM-2 embedding backend — the formal fallback/comparison pipeline (blueprint §7).

Note: ESM-2 operates on amino-acid sequences, not raw nucleotides — unlike
Genos-m it is a protein-level model. Callers comparing the two on the same
material must pass translated/predicted ORFs to this backend and the
corresponding nucleotide sequences to GenosMEmbedder.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from darkmatter.device import from_pretrained_kwargs, inference_device, plan_for_model
from darkmatter.embeddings.base import Embedder


class ESM2Embedder(Embedder):
    name = "esm2"

    def __init__(self, hf_repo: str, vram_bf16_gb: float, quantize_below_vram_gb: float, max_tokens: int = 1024,
                 revision: str | None = None):
        self.plan = plan_for_model(vram_bf16_gb, quantize_below_vram_gb)
        self.max_tokens = max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo, revision=revision)

        self.model = AutoModel.from_pretrained(hf_repo, revision=revision, **from_pretrained_kwargs(self.plan))
        if not self.plan.quantized:
            self.model.to(self.plan.device)
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size

    @torch.inference_mode()
    def embed(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        out = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            tokens = {k: v.to(inference_device(self.plan, self.model)) for k, v in tokens.items()}

            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())

        return np.concatenate(out, axis=0)
