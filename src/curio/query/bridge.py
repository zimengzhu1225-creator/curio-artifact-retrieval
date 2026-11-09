"""Cross-lingual query bridge: offline MT plus score-level fusion.

Motivation (measured, not assumed): on the frozen visual split the same 24
semantic intents score much lower in Chinese than in English.  Replacing a
Chinese query by its human English counterpart raises ``eval/visual_test``
nDCG@10 from 0.8264 to 0.9152, so the deficit is a *text-channel* effect rather
than a misunderstanding of the request.  Translating the Chinese query and
fusing the two dense channels recovers part of that gap without touching the
image index, the LoRA adapters, or the frozen catalogue.

Design constraints kept from the repository:

* Offline only - the translation model is loaded from a pinned local snapshot;
  no API keys, no per-request network access, no LLM planner.
* The image side is untouched: identical embeddings, identical catalogue.
* The fusion weight and the translated-channel form are selected on
  ``eval/visual_dev`` only; ``eval/visual_test`` is used once afterwards.
* English queries are never rewritten by this bridge; it is identity for them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-zh-en"
NLLB_SOURCE = "zho_Hans"
NLLB_TARGET = "eng_Latn"


def _is_nllb(model_name: str) -> bool:
    """NLLB checkpoints need language tokens; Marian checkpoints do not."""
    return "nllb" in Path(str(model_name)).name.lower()


def zscore(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    deviation = float(values.std())
    return (values - float(values.mean())) / (deviation if deviation > 1e-8 else 1.0)


def fuse_scores(primary: np.ndarray, bridge: np.ndarray, weight: float) -> np.ndarray:
    """Fuse two same-scale channels; ``weight=0`` is exactly the primary channel."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    if np.asarray(primary).shape != np.asarray(bridge).shape:
        raise ValueError("Both channels must score the same candidate pool")
    if weight == 0.0:
        return np.asarray(primary, dtype=np.float32)
    return ((1.0 - weight) * zscore(primary) + weight * zscore(bridge)).astype(np.float32)


def needs_bridge(parsed) -> bool:
    """Only Chinese input is bridged; English already sits in the strong channel."""
    return str(getattr(parsed, "language", "")).lower().startswith("zh")


@dataclass(frozen=True)
class BridgeConfig:
    model_name: str = DEFAULT_MODEL
    weight: float = 0.0
    channel: str = "rewritten"   # "raw" | "rewritten" | "plus_slots"
    terminology: bool = False    # append concepts the translation dropped

    def to_dict(self) -> dict:
        return {"model_name": self.model_name, "weight": self.weight,
                "channel": self.channel, "terminology": self.terminology}


class TranslationCache:
    """Append-only translation cache keyed by source text, model, and channel."""

    def __init__(self, path: str | Path | None = None, model_name: str = DEFAULT_MODEL):
        self.path = Path(path) if path else None
        self.model_name = model_name
        self.entries: dict[str, str] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("model") == model_name:
                    self.entries[row["sha256"]] = row["translation"]

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> str | None:
        return self.entries.get(self.key(text))

    def put(self, text: str, translation: str, *, persist: bool = True) -> None:
        self.entries[self.key(text)] = translation
        if persist and self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"sha256": self.key(text), "model": self.model_name,
                                         "source": text, "translation": translation},
                                        ensure_ascii=False) + "\n")


class CrossLingualBridge:
    """Lazy offline translator; ``translate`` never touches the network twice."""

    def __init__(self, config: BridgeConfig, cache: TranslationCache | None = None):
        self.config = config
        self.cache = cache or TranslationCache(model_name=config.model_name)
        self._tokenizer = None
        self._model = None
        self.translation_ms = 0.0

    def _load(self):
        if self._model is None:
            if _is_nllb(self.config.model_name):
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, src_lang=NLLB_SOURCE)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name).eval()
            else:
                from transformers import MarianMTModel, MarianTokenizer

                self._tokenizer = MarianTokenizer.from_pretrained(self.config.model_name)
                self._model = MarianMTModel.from_pretrained(self.config.model_name).eval()
        return self._tokenizer, self._model

    def translate(self, texts: Iterable[str], *, batch_size: int = 16) -> list[str]:
        import time
        import torch

        texts = list(texts)
        pending = [text for text in texts if self.cache.get(text) is None]
        if pending:
            tokenizer, model = self._load()
            started = time.perf_counter()
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
                extra = {}
                if _is_nllb(self.config.model_name):
                    extra["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(NLLB_TARGET)
                with torch.inference_mode():
                    output = model.generate(**encoded, max_new_tokens=96, num_beams=2, **extra)
                for source, tokens in zip(batch, output):
                    self.cache.put(source, tokenizer.decode(tokens, skip_special_tokens=True))
            self.translation_ms += (time.perf_counter() - started) * 1000
        return [self.cache.get(text) for text in texts]

    def bridge_text(self, parsed, translation: str, channel: str | None = None,
                    terminology: bool | None = None) -> str:
        """Compose the translated channel from the frozen parser, not free text."""
        from curio.query import parse_query
        from curio.query.terminology import repair_translation

        channel = channel or self.config.channel
        use_terminology = self.config.terminology if terminology is None else terminology
        if use_terminology:
            translation = repair_translation(getattr(parsed, "query", "") or "", translation,
                                             getattr(parsed, "slots", None))
        if channel == "raw":
            return translation
        rewritten = parse_query(translation).rewritten
        if channel == "rewritten":
            return rewritten
        if channel == "plus_slots":
            return f"{translation} {parsed.rewritten}".strip()
        raise ValueError(f"Unknown channel {channel!r}")
