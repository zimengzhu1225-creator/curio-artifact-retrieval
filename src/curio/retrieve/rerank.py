"""Contradiction-gated attribute reranking over dense image candidates.

The production ``image_only`` route in :mod:`curio.retrieve.engine` adds a
softmax probe bonus to every candidate. Frozen visual-dev selection shows that
this naive fusion loses nDCG@10 monotonically as the weight grows
(0.9175 -> 0.9086 -> 0.8801 -> 0.8415), so the production weight stays 0.

This module implements a different mechanism, kept deliberately separate from
the default route:

1. **Gate** - only the dense top-N band may be reordered. Candidates outside
   the band keep their dense order and can never be promoted, which removes the
   "high probe probability climbs from rank 50" failure mode.
2. **Conjunctive support** - a candidate must satisfy *all* covered query
   attributes, so ``min`` over slot supports is available instead of the mean
   that lets a strong object-type match hide a wrong colour.
3. **Contradiction penalty** - a candidate whose probe confidently prefers a
   different class for a stated attribute is pushed down by the margin above a
   threshold, instead of merely receiving no bonus.

All outputs remain weak-supervision ranking signals, never catalogue facts or
human relevance judgments.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

RERANK_MODES = ("none", "naive", "gate")
AGGREGATIONS = ("mean", "min")


@dataclass(frozen=True)
class RerankConfig:
    """Frozen reranker configuration; every field is selected on dev only."""

    mode: str = "gate"
    top_n: int = 30
    support_weight: float = 0.0
    conflict_weight: float = 0.0
    conflict_margin: float = 0.0
    aggregation: str = "min"

    def __post_init__(self):
        if self.mode not in RERANK_MODES:
            raise ValueError(f"mode must be one of {RERANK_MODES}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
        if self.top_n < 1:
            raise ValueError("top_n must be positive")
        for name in ("support_weight", "conflict_weight", "conflict_margin"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite float in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SlotEvidence:
    """Per-slot probabilities for one candidate set."""

    task: str
    value: str
    negated: bool
    support: np.ndarray
    other: np.ndarray


def _class_index(classes: list[str], value: str) -> int | None:
    return classes.index(value) if value in classes else None


def slot_evidence(parsed: Any, predictions: dict, rows: np.ndarray) -> list[SlotEvidence]:
    """Return covered query-attribute evidence for the rows in a candidate pool.

    ``predictions`` is ``{task: {"classes": [...], "probabilities": (n, c)}}``.
    Uncovered attributes are omitted, never treated as absent.
    """
    rows = np.asarray(rows)
    evidence: list[SlotEvidence] = []
    for task, values in parsed.slots.items():
        prediction = predictions.get(task)
        if not prediction:
            continue
        classes = list(prediction["classes"])
        probabilities = np.asarray(prediction["probabilities"], dtype=np.float32)
        for value in values:
            column = _class_index(classes, value)
            if column is None:
                continue
            block = probabilities[rows]
            support = block[:, column]
            if len(classes) > 1:
                others = np.delete(block, column, axis=1).max(axis=1)
            else:
                others = np.zeros_like(support)
            evidence.append(SlotEvidence(task, value, False, support, others))
    seen: set[tuple[str, str]] = set()
    for negative in getattr(parsed, "negations", []) or []:
        task, value = negative["slot"], negative["value"]
        if (task, value) in seen:
            continue
        seen.add((task, value))
        prediction = predictions.get(task)
        if not prediction:
            continue
        classes = list(prediction["classes"])
        column = _class_index(classes, value)
        if column is None:
            continue
        block = np.asarray(prediction["probabilities"], dtype=np.float32)[rows]
        probability = block[:, column]
        others = np.delete(block, column, axis=1).max(axis=1) if len(classes) > 1 else np.zeros_like(probability)
        evidence.append(SlotEvidence(task, value, True, 1.0 - probability, probability - (1.0 - probability)))
    return evidence


def attribute_scores(evidence: list[SlotEvidence], *, aggregation: str = "min") -> tuple[np.ndarray, np.ndarray]:
    """Aggregate slot evidence into support and raw conflict-margin arrays."""
    if not evidence:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    support = np.vstack([item.support for item in evidence])
    margins = np.vstack([np.maximum(0.0, item.other - item.support) for item in evidence])
    aggregate = support.min(axis=0) if aggregation == "min" else support.mean(axis=0)
    return aggregate.astype(np.float32), margins.mean(axis=0).astype(np.float32)


def rank_normalized(scores: np.ndarray) -> np.ndarray:
    """RRF-style 61/(60+rank) normalization, matching the production engine."""
    order = np.argsort(-scores, kind="stable")
    normalized = np.empty(len(scores), dtype=np.float32)
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float32)
    normalized[:] = 61.0 / (60.0 + ranks)
    return normalized


def rerank_pool(
    dense_scores: np.ndarray,
    support: np.ndarray | None,
    conflict: np.ndarray | None,
    config: RerankConfig,
) -> tuple[np.ndarray, dict]:
    """Reorder one candidate pool; returns the new order (indices into the pool).

    ``support`` and ``conflict`` must align with ``dense_scores`` and may be
    ``None`` when the query has no covered attributes (the pool then keeps its
    dense order and ``applied`` is ``False``).
    """
    scores = np.asarray(dense_scores, dtype=np.float32)
    total = len(scores)
    order = list(np.argsort(-scores, kind="stable"))
    diagnostics = {
        "mode": config.mode,
        "candidates": total,
        "eligible": 0,
        "applied": False,
        "mean_support": None,
        "mean_conflict": None,
        "moved": 0,
    }
    if config.mode == "none" or total < 2:
        return np.asarray(order), diagnostics
    if config.mode == "naive":
        if support is None or not len(support):
            return np.asarray(order), diagnostics
        fused = rank_normalized(scores) + config.support_weight * np.asarray(support, dtype=np.float32)
        order = list(np.argsort(-fused, kind="stable"))
        diagnostics.update({"eligible": total, "applied": True,
                            "mean_support": float(np.mean(support)),
                            "mean_conflict": float(np.mean(conflict)) if conflict is not None and len(conflict) else None,
                            "moved": int(np.sum(np.asarray(order) != np.arange(total)))})
        return np.asarray(order), diagnostics
    if support is None or not len(support):
        return np.asarray(order), diagnostics
    band = min(config.top_n, total)
    band_indices = np.asarray(order[:band])
    band_scores = rank_normalized(scores[band_indices])
    band_scores = band_scores + config.support_weight * (np.asarray(support, dtype=np.float32)[band_indices] - 0.5)
    if conflict is not None and len(conflict) and config.conflict_weight:
        penalty = np.maximum(0.0, np.asarray(conflict, dtype=np.float32)[band_indices] - config.conflict_margin)
        band_scores = band_scores - config.conflict_weight * penalty
    band_order = np.argsort(-band_scores, kind="stable")
    new_order = np.concatenate([band_indices[band_order], np.asarray(order[band:], dtype=np.int64)])
    diagnostics.update({"eligible": band, "applied": True,
                        "mean_support": float(np.mean(np.asarray(support)[band_indices])),
                        "mean_conflict": float(np.mean(np.asarray(conflict)[band_indices])) if conflict is not None and len(conflict) else None,
                        "moved": int(np.sum(new_order != np.asarray(order)))})
    return new_order, diagnostics


def rerank_ranking(
    dense_scores: np.ndarray,
    parsed: Any,
    predictions: dict,
    rows: np.ndarray,
    config: RerankConfig,
) -> tuple[list[int], dict]:
    """Rank an explicit candidate pool (e.g. a restricted gallery).

    ``rows`` maps pool positions to rows of the probe probability matrices.
    Returns the item-row order plus diagnostics for audit output.
    """
    evidence = slot_evidence(parsed, predictions, rows)
    support, conflict = attribute_scores(evidence, aggregation=config.aggregation)
    order, diagnostics = rerank_pool(dense_scores, support if evidence else None,
                                     conflict if evidence else None, config)
    diagnostics["covered_slots"] = [f"{item.task}={item.value}{' (not)' if item.negated else ''}" for item in evidence]
    diagnostics["support_aggregation"] = config.aggregation
    return [int(rows[position]) for position in order], diagnostics
