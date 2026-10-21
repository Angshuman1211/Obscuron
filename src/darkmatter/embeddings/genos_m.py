"""Genos-m embedding backend.

Genos-m (BGI-Research / Zhejiang Lab, Apache 2.0) is a sparse MoE DNA
foundation model: 4.7B total / ~330M active params, single-nucleotide
tokenization, mean-pooled sequence embeddings. See
https://huggingface.co/BGI-HangzhouAI/Genos-m

Full bf16 weights (~9GB) exceed 8GB-class consumer GPUs, so this backend
quantizes to 8-bit automatically below the VRAM threshold in
config/models.yaml (see darkmatter.device.plan_for_model).
"""

from __future__ import annotations

import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from darkmatter.device import from_pretrained_kwargs, inference_device, plan_for_model
from darkmatter.embeddings.base import Embedder

_RUN_OF_N = re.compile(r"N{3,}")


def _preprocess(seq: str) -> str:
    """Match Genos-m's expected preprocessing: runs of 3+ N -> '@'."""
    return _RUN_OF_N.sub("@", seq.upper())


class GenosMEmbedder(Embedder):
    name = "genos-m"

    def __init__(
        self,
        hf_repo: str,
        vram_bf16_gb: float,
        quantize_below_vram_gb: float,
        max_tokens: int = 8192,
        layer: int | None = None,
        revision: str | None = None,
    ):
        self.plan = plan_for_model(vram_bf16_gb, quantize_below_vram_gb)
        self.max_tokens = max_tokens
        self.layer = layer

        # trust_remote_code runs the repo's own Python, so it is pinned to a reviewed commit
        # (config/models.yaml revision) rather than whatever the repo's main branch holds today
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True, revision=revision)

        load_kwargs = {"trust_remote_code": True, "revision": revision, **from_pretrained_kwargs(self.plan)}
        self.model = AutoModel.from_pretrained(hf_repo, **load_kwargs)
        if not self.plan.quantized:
            self.model.to(self.plan.device)
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size

    @torch.inference_mode()
    def embed(self, sequences: list[str], batch_size: int = 4) -> np.ndarray:
        out = []
        for i in range(0, len(sequences), batch_size):
            batch = [_preprocess(s) for s in sequences[i : i + batch_size]]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            tokens = {k: v.to(inference_device(self.plan, self.model)) for k, v in tokens.items()}

            if self.layer is None:
                hidden = self.model(**tokens).last_hidden_state  # (B, T, H)
            else:
                hidden = self.model(**tokens, output_hidden_states=True).hidden_states[self.layer]
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())

        return np.concatenate(out, axis=0)
