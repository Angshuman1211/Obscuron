"""ProstT5 embedding backend -- Layer 2's structure-aware arm (blueprint
Layer 2 / P4-D6). ProtT5-XL fine-tuned to translate between amino-acid
sequence and Foldseek 3Di structural tokens, used encoder-only: no 3-D
folding, so this runs like any other sequence embedder while still
carrying structure-informed representation, the lightweight ESMFold
substitute the available 8GB-class hardware requires.

Two GOTCHAs this module works around (P4-D6):
  (i) transformers loads safetensors only; ProstT5 ships a plain fp16
      pytorch_model.bin (in a separate weights repo) -- load it by hand
      into a T5EncoderModel built from the tokenizer/config repo.
  (ii) ProtT5-family tokenizers expect space-separated residues with rare
      amino acids (U, Z, O, B) mapped to X, not a raw sequence string.
"""

from __future__ import annotations

import re

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import T5Config, T5EncoderModel, T5Tokenizer

from darkmatter.device import inference_device, plan_for_model
from darkmatter.embeddings.base import Embedder

_RARE_AA = re.compile(r"[UZOB]")


def _preprocess(seq: str) -> str:
    return " ".join(_RARE_AA.sub("X", seq.upper()))


class ProstT5Embedder(Embedder):
    name = "prostt5"

    def __init__(self, hf_repo: str, vram_bf16_gb: float, quantize_below_vram_gb: float,
                max_tokens: int = 1024, weights_repo: str | None = None,
                weights_file: str = "pytorch_model.bin", revision: str | None = None,
                weights_revision: str | None = None):
        self.plan = plan_for_model(vram_bf16_gb, quantize_below_vram_gb)
        self.max_tokens = max_tokens

        self.tokenizer = T5Tokenizer.from_pretrained(hf_repo, do_lower_case=False, revision=revision)
        config = T5Config.from_pretrained(hf_repo, revision=revision)
        target_dtype = torch.float16 if not self.plan.quantized else torch.float32
        target_device = self.plan.device
        # Build directly on the target device in the target dtype -- never
        # allocate a full fp32 CPU copy, and never do the recursive
        # per-parameter .to(device) transfer (hundreds of small CUDA
        # allocations from that call is what was fragmenting the allocator
        # badly enough to OOM on a spare 32MiB with 5GiB nominally free).
        with torch.device("meta"):
            self.model = T5EncoderModel(config)
        self.model = self.model.to_empty(device=target_device).to(target_dtype)

        weights_src = weights_repo or hf_repo
        weights_path = hf_hub_download(repo_id=weights_src, filename=weights_file,
                                       revision=weights_revision if weights_repo else revision)
        state_dict = torch.load(weights_path, map_location=target_device, weights_only=True)
        # the checkpoint is the full seq2seq model; keep only the encoder's weights
        encoder_state = {k[len("encoder."):]: v.to(target_dtype)
                         for k, v in state_dict.items() if k.startswith("encoder.")}
        missing, unexpected = self.model.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"[prostt5] missing keys (ok if just a handful of buffers): {missing[:5]}", flush=True)
        del state_dict, encoder_state

        self.model.eval()

        self.embedding_dim = config.d_model

    @torch.inference_mode()
    def embed(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        device = inference_device(self.plan, self.model)
        out = []
        for i in range(0, len(sequences), batch_size):
            batch = [_preprocess(s) for s in sequences[i:i + batch_size]]
            tokens = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_tokens, return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())

        return np.concatenate(out, axis=0)
