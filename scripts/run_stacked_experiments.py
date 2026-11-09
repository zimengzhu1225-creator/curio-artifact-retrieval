"""Measure the cross-lingual bridge on top of the compositional retriever.

Systems scored on the same frozen gallery, so the numbers are directly comparable:

1. ``baseline``      - single dense vector from the parsed Chinese rewrite.
2. ``subquery_and``  - teammate @zhizhizhi-victor's clause decomposition + Soft-AND.
3. ``bridge``        - cross-lingual bridge alone (translation + terminology repair).
4. ``stacked``       - bridge channel fused on top of ``subquery_and``.

The fusion weight is selected on ``visual_dev`` and frozen before the single
``visual_test`` evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from curio.evaluate.metrics import evaluate_ranking
from curio.evaluate.stats import bootstrap_ci, paired_permutation
from curio.query import parse_query
from curio.query.bridge import BridgeConfig, CrossLingualBridge, TranslationCache, zscore
from curio.represent.encoders import Encoder
from curio.retrieve.compositional import (
    DEFAULT_LAMBDA_AND,
    DEFAULT_LAMBDA_NEG,
    DEFAULT_POOL_K,
    DEFAULT_TAU,
    decompose_visual_query,
    soft_and_zscores,
)

METRICS = ("ndcg@10", "precision@5", "ap@10", "mrr@10")


def read_jsonl(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_split(name: str) -> dict:
    root = Path(f"eval/visual_{name}")
    queries = read_jsonl(root / "queries.jsonl")
    qrels = {row["qid"]: row["grades"] for row in read_jsonl(root / "qrels.jsonl")}
    gallery = [str(key) for key in json.loads((root / "gallery_ids.json").read_text(encoding="utf-8"))]
    return {"queries": queries, "qrels": qrels, "gallery": gallery}


def load_index() -> tuple[dict[str, int], np.ndarray, dict]:
    directory = Path("artifacts/siglip_adapted")
    ids = json.loads((directory / "item_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    values = np.load(directory / "image_embeddings.npy", allow_pickle=False).astype(np.float32)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    return {key: index for index, key in enumerate(ids)}, values, manifest


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def compositional_scores(cand_embs: np.ndarray, q_vecs: np.ndarray, decomp: dict) -> np.ndarray:
    """Reimplementation of the teammate's ``subquery_and`` formula (engine.py)."""
    s0 = cand_embs @ q_vecs[0]
    pool_idx = np.argsort(-s0)[:min(DEFAULT_POOL_K, len(s0))]
    deviation = float(s0[pool_idx].std())
    scale = deviation if deviation > 1e-5 else 0.05
    combined = s0.astype(np.float64).copy()
    n_sub, n_neg = len(decomp["sub_prompts"]), len(decomp["neg_prompts"])
    if n_sub >= 2:
        sub_mat = cand_embs @ q_vecs[1:1 + n_sub].T
        combined += DEFAULT_LAMBDA_AND * scale * soft_and_zscores(sub_mat, pool_idx, tau=DEFAULT_TAU)
    if n_neg >= 1:
        neg_mat = cand_embs @ q_vecs[1 + n_sub:1 + n_sub + n_neg].T
        pool_neg = neg_mat[pool_idx]
        z_neg = np.clip(
            (neg_mat - pool_neg.mean(axis=0, keepdims=True))
            / np.maximum(pool_neg.std(axis=0, keepdims=True), 1e-4),
            0.0,
            3.5,
        ).max(axis=1)
        combined -= DEFAULT_LAMBDA_NEG * scale * z_neg
    return combined


def rank_metrics(scores: np.ndarray, gallery: list[str], grades: dict) -> dict:
    order = np.argsort(-scores, kind="stable")
    return evaluate_ranking([gallery[i] for i in order], grades)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for key in METRICS:
        values = [row[key] for row in rows]
        interval = bootstrap_ci(values)
        out[key] = {"mean": float(np.mean(values)),
                    "lower": float(interval["lower"]),
                    "upper": float(interval["upper"])}
    return out


def build_channels(split: dict, encoder, bridge, position, vectors) -> list[dict]:
    """Encode every prompt variant for one split in two batched passes."""
    gallery = split["gallery"]
    cand_embs = vectors[np.array([position[key] for key in gallery])]
    records, batch, en_batch = [], [], []
    for query in split["queries"]:
        parsed = parse_query(query["text_zh"])
        decomp = decompose_visual_query(query["text_zh"], parsed)
        translation = bridge.translate([query["text_zh"]])[0]
        prompts = [decomp["global_prompt"], *decomp["sub_prompts"], *decomp["neg_prompts"]]
        records.append({"qid": query["qid"], "text": query["text_zh"], "parsed": parsed,
                        "decomp": decomp, "translation": translation,
                        "start": len(batch), "stop": len(batch) + len(prompts)})
        batch.extend(prompts)
        en_batch.append(bridge.bridge_text(parsed, translation, "rewritten", True))
    prompts_matrix = unit(np.asarray(encoder.encode_texts(batch), dtype=np.float32))
    en_matrix = unit(np.asarray(encoder.encode_texts(en_batch), dtype=np.float32))
    rewrite_matrix = unit(np.asarray(encoder.encode_texts([r["parsed"].rewritten for r in records]),
                                     dtype=np.float32))
    for index, record in enumerate(records):
        q_vecs = prompts_matrix[record["start"]:record["stop"]]
        record["baseline"] = cand_embs @ rewrite_matrix[index]
        record["subquery_and"] = compositional_scores(cand_embs, q_vecs, record["decomp"])
        record["bridge"] = cand_embs @ en_matrix[index]
        record["n_sub"] = len(record["decomp"]["sub_prompts"])
        record["n_neg"] = len(record["decomp"]["neg_prompts"])
    return records


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--out", default="artifacts/stacked_eval")
    cli.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    cli.add_argument("--model", default="artifacts/models/nllb-200-distilled-600M")
    cli.add_argument("--cache", default="artifacts/bridge_2x2/translations.jsonl")
    cli.add_argument("--no-terminology", dest="terminology", action="store_false")
    cli.set_defaults(terminology=True)
    args = cli.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    weights = [float(item) for item in args.weights.split(",")]

    position, vectors, manifest = load_index()
    encoder = Encoder("siglip", "auto", adapter_path=manifest.get("adapter_path"))
    cache = TranslationCache(args.cache, model_name=args.model)
    bridge = CrossLingualBridge(BridgeConfig(model_name=args.model, terminology=args.terminology), cache)

    started = time.perf_counter()
    payload: dict = {
        "config": {"model": args.model, "terminology": args.terminology,
                   "lambda_and": DEFAULT_LAMBDA_AND, "lambda_neg": DEFAULT_LAMBDA_NEG, "tau": DEFAULT_TAU},
        "splits": {},
    }

    curves: dict[str, dict[float, list[dict]]] = {}
    qid_order: dict[str, list[str]] = {}
    for name in ("dev", "test"):
        split = load_split(name)
        records = build_channels(split, encoder, bridge, position, vectors)
        qid_order[name] = [record["qid"] for record in records]
        gallery, qrels = split["gallery"], split["qrels"]
        payload["splits"][name] = {
            "single": {
                key: summarize([rank_metrics(record[key], gallery, qrels[record["qid"]]) for record in records])
                for key in ("baseline", "subquery_and", "bridge")
            },
            "per_query": {
                record["qid"]: {
                    "text": record["text"], "n_sub": record["n_sub"], "n_neg": record["n_neg"],
                    **{key: rank_metrics(record[key], gallery, qrels[record["qid"]])
                       for key in ("baseline", "subquery_and", "bridge")},
                } for record in records
            },
        }
        curves[name] = {}
        for weight in weights:
            rows = []
            for record in records:
                fused = (1 - weight) * zscore(record["subquery_and"]) + weight * zscore(record["bridge"])
                rows.append(rank_metrics(fused, gallery, qrels[record["qid"]]))
            curves[name][weight] = rows
        payload["splits"][name]["fusion_curve"] = {
            f"{weight:.1f}": summarize(rows) for weight, rows in curves[name].items()
        }

    dev_curve = payload["splits"]["dev"]["fusion_curve"]
    best = max(dev_curve, key=lambda key: dev_curve[key]["ndcg@10"]["mean"])
    frozen_weight = float(best)
    test_rows = curves["test"][frozen_weight]
    # Paired comparison against the teammate's retriever at the frozen weight.
    reference = payload["splits"]["test"]["per_query"]
    paired = paired_permutation(
        [reference[qid]["subquery_and"]["ndcg@10"] for qid in sorted(reference)],
        [row["ndcg@10"] for row in test_rows],
    )
    payload["selection"] = {
        "weight": frozen_weight,
        "dev_ndcg": dev_curve[best]["ndcg@10"],
        "test_frozen": summarize(test_rows),
        "paired_vs_subquery_and": paired,
        "test_per_query": [
            {"qid": qid,
             "text": reference[qid]["text"],
             "n_sub": reference[qid]["n_sub"],
             "subquery_and": reference[qid]["subquery_and"]["ndcg@10"],
             "stacked": float(row["ndcg@10"]),
             "delta": float(row["ndcg@10"] - reference[qid]["subquery_and"]["ndcg@10"])}
            for qid, row in zip(qid_order["test"], test_rows)
        ],
    }
    payload["elapsed_s"] = round(time.perf_counter() - started, 1)
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Stacked evaluation: cross-lingual bridge on top of subquery_and", "",
             f"Translation model: `{args.model}` | terminology repair: **{args.terminology}**", "",
             f"Fusion weight selected on dev: **w={frozen_weight:.1f}**", ""]
    for name in ("dev", "test"):
        block = payload["splits"][name]
        lines.append(f"## {'dev (weight selection)' if name == 'dev' else 'test (single frozen run)'}")
        lines.append("")
        lines.append("| System | nDCG@10 (95% CI) | P@5 | AP@10 | MRR@10 |")
        lines.append("|---|---:|---:|---:|---:|")
        for key, label in (("baseline", "baseline (single vector)"),
                           ("subquery_and", "subquery_and (teammate)"),
                           ("bridge", "bridge only (ours)")):
            metrics = block["single"][key]
            lines.append(
                f"| {label} | {metrics['ndcg@10']['mean']:.4f} "
                f"[{metrics['ndcg@10']['lower']:.4f}, {metrics['ndcg@10']['upper']:.4f}] | "
                f"{metrics['precision@5']['mean']:.4f} | {metrics['ap@10']['mean']:.4f} | "
                f"{metrics['mrr@10']['mean']:.4f} |"
            )
        lines.append("")
        lines.append("| Fusion weight w | nDCG@10 | P@5 | AP@10 | MRR@10 |")
        lines.append("|---|---:|---:|---:|---:|")
        for weight, metrics in block["fusion_curve"].items():
            marker = " **(selected)**" if name == "dev" and weight == best else ""
            lines.append(f"| {weight}{marker} | {metrics['ndcg@10']['mean']:.4f} | "
                         f"{metrics['precision@5']['mean']:.4f} | {metrics['ap@10']['mean']:.4f} | "
                         f"{metrics['mrr@10']['mean']:.4f} |")
        lines.append("")
    lines.append(f"Paired permutation (stacked vs subquery_and, test): "
                 f"mean delta {paired['difference']['mean']:+.4f} "
                 f"[{paired['difference']['lower']:+.4f}, {paired['difference']['upper']:+.4f}], "
                 f"p={paired['p_value']:.4f}")
    lines.append("")
    lines.append("> Relevance labels are the disclosed AI visual-audit judgments, not independent human ground truth.")
    lines.append("> 24 Chinese queries per split; a positive point estimate is not by itself significance.")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[done] {payload['elapsed_s']}s")


if __name__ == "__main__":
    main()
