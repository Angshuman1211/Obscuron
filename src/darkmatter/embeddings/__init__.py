from __future__ import annotations

from darkmatter.config import models_config
from darkmatter.embeddings.base import Embedder

_BACKENDS = ("genos-m", "esm2", "prostt5")


def load_embedder(backend: str, layer: int | None = None) -> Embedder:
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}, expected one of {_BACKENDS}")

    cfg = models_config()

    if backend == "prostt5":
        from darkmatter.embeddings.prostt5 import ProstT5Embedder

        m = cfg["prostt5"]
        return ProstT5Embedder(
            hf_repo=m["hf_repo"],
            vram_bf16_gb=m["vram_bf16_gb"],
            quantize_below_vram_gb=m["quantize_below_vram_gb"],
            max_tokens=m["max_tokens"],
            weights_repo=m.get("weights_repo"),
            weights_file=m.get("weights_file", "pytorch_model.bin"),
            revision=m.get("revision"),
            weights_revision=m.get("weights_revision"),
        )

    if backend == "genos-m":
        from darkmatter.embeddings.genos_m import GenosMEmbedder

        m = cfg["genos_m"]
        return GenosMEmbedder(
            hf_repo=m["hf_repo"],
            vram_bf16_gb=m["vram_bf16_gb"],
            quantize_below_vram_gb=m["quantize_below_vram_gb"],
            max_tokens=m["max_tokens"],
            layer=layer,
            revision=m.get("revision"),
        )

    from darkmatter.embeddings.esm2 import ESM2Embedder

    m = cfg["esm2"]
    return ESM2Embedder(
        hf_repo=m["hf_repo"],
        vram_bf16_gb=m["vram_bf16_gb"],
        quantize_below_vram_gb=m["quantize_below_vram_gb"],
        max_tokens=m["max_tokens"],
        revision=m.get("revision"),
    )
