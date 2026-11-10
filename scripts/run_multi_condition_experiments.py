"""Evaluate the frozen systems on the multi-condition subset (eval/multi_test).

The subset holds 20 Chinese queries that each carry at least two independent
conditions separated by a comma, so the compositional clause splitter actually
fires. No parameter is re-tuned here: the fusion weight is the one frozen on
visual_dev (w=0.4 default from the stacked experiment).

Reported systems (identical implementations to run_stacked_experiments.py):
1. baseline      - single dense vector from the parsed Chinese rewrite.
2. subquery_and  - teammate's clause decomposition + Soft-AND.
3. bridge        - cross-lingual bridge alone.
4. stacked       - (1-w)*z(subquery_and) + w*z(bridge), frozen w.

Primary comparisons (paired permutation on nDCG@10):
- subquery_and vs baseline   -> does Soft-AND pay off on multi-condition queries?
- stacked vs subquery_and    -> does the bridge add on top of Soft-AND?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from curio.evaluate.stats import bootstrap_ci, paired_permutation
from curio.evaluate.metrics import evaluate_ranking
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


def load_multi_split(root: str = "eval/multi_test") -> dict:
    root = Path(root)
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
    cli.add_argument("--out", default="artifacts/multi_condition_eval")
    cli.add_argument("--split", default="eval/multi_test")
    cli.add_argument("--weight", type=float, default=0.4,
                     help="frozen fusion weight selected on visual_dev")
    cli.add_argument("--model", default="artifacts/models/nllb-200-distilled-600M")
    cli.add_argument("--cache", default="artifacts/bridge_2x2/translations.jsonl")
    args = cli.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    position, vectors, manifest = load_index()
    encoder = Encoder("siglip", "auto", adapter_path=manifest.get("adapter_path"))
    cache = TranslationCache(args.cache, model_name=args.model)
    bridge = CrossLingualBridge(BridgeConfig(model_name=args.model, terminology=True), cache)

    started = time.perf_counter()
    split = load_multi_split(args.split)
    records = build_channels(split, encoder, bridge, position, vectors)
    gallery, qrels = split["gallery"], split["qrels"]

    for record in records:
        fused = (1 - args.weight) * zscore(record["subquery_and"]) + args.weight * zscore(record["bridge"])
        record["stacked"] = fused

    payload: dict = {
        "config": {"split": args.split, "frozen_weight": args.weight, "model": args.model,
                   "terminology": True, "lambda_and": DEFAULT_LAMBDA_AND,
                   "lambda_neg": DEFAULT_LAMBDA_NEG, "tau": DEFAULT_TAU},
        "subset": {"n_queries": len(records),
                   "n_sub_distribution": {str(n): sum(1 for r in records if r["n_sub"] == n)
                                          for n in sorted({r["n_sub"] for r in records})},
                   "soft_and_triggered": sum(1 for r in records if r["n_sub"] >= 2)},
    }
    payload["single"] = {
        key: summarize([rank_metrics(record[key], gallery, qrels[record["qid"]]) for record in records])
        for key in ("baseline", "subquery_and", "bridge", "stacked")
    }
    payload["per_query"] = {
        record["qid"]: {
            "text": record["text"], "n_sub": record["n_sub"], "n_neg": record["n_neg"],
            **{key: rank_metrics(record[key], gallery, qrels[record["qid"]])
               for key in ("baseline", "subquery_and", "bridge", "stacked")},
        } for record in records
    }

    qids = [r["qid"] for r in records]
    def col(key: str) -> list[float]:
        return [payload["per_query"][qid][key]["ndcg@10"] for qid in qids]
    payload["paired_tests"] = {
        "subquery_and_vs_baseline": paired_permutation(col("baseline"), col("subquery_and")),
        "stacked_vs_subquery_and": paired_permutation(col("subquery_and"), col("stacked")),
        "bridge_vs_baseline": paired_permutation(col("baseline"), col("bridge")),
    }

    payload["elapsed_s"] = round(time.perf_counter() - started, 1)
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Multi-condition subset: frozen systems on eval/multi_test", "",
             f"Subset: {len(records)} Chinese queries, >=2 conditions each (comma-separated) | "
             f"fusion weight frozen at w={args.weight} | no re-tuning", "",
             f"Soft-AND clause split triggered on **{payload['subset']['soft_and_triggered']}/{len(records)}** queries "
             f"(n_sub distribution: {payload['subset']['n_sub_distribution']})", "",
             "| System | nDCG@10 (95% CI) | P@5 | AP@10 | MRR@10 |",
             "|---|---:|---:|---:|---:|"]
    labels = (("baseline", "baseline (single vector)"),
              ("subquery_and", "subquery_and (teammate)"),
              ("bridge", "bridge only (ours)"),
              ("stacked", "stacked (bridge on subquery_and)"))
    for key, label in labels:
        m = payload["single"][key]["ndcg@10"]
        lines.append(f"| {label} | {m['mean']:.4f} [{m['lower']:.4f}, {m['upper']:.4f}] | "
                     f"{payload['single'][key]['precision@5']['mean']:.4f} | "
                     f"{payload['single'][key]['ap@10']['mean']:.4f} | "
                     f"{payload['single'][key]['mrr@10']['mean']:.4f} |")
    lines.append("")
    for name, test in payload["paired_tests"].items():
        d = test["difference"]
        lines.append(f"- {name}: mean delta {d['mean']:+.4f} "
                     f"[{d['lower']:+.4f}, {d['upper']:+.4f}], p={test['p_value']:.4f}")
    lines.append("")
    lines.append("| qid | query | n_sub | base | sub_and | bridge | stacked |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for qid in qids:
        row = payload["per_query"][qid]
        lines.append(f"| {qid} | {row['text'][:28]}… | {row['n_sub']} | "
                     f"{row['baseline']['ndcg@10']:.4f} | {row['subquery_and']['ndcg@10']:.4f} | "
                     f"{row['bridge']['ndcg@10']:.4f} | {row['stacked']['ndcg@10']:.4f} |")
    lines.append("")
    lines.append("> Labels are conjunction-derived with AI visual re-judgment of base->=1 images; not human gold.")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[done] {payload['elapsed_s']}s")


if __name__ == "__main__":
    main()
