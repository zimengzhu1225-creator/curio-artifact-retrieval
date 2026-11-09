"""2x2 ablation: translation model x terminology repair.

Cells
-----
``opus``  Helsinki-NLP/opus-mt-zh-en (74M, the earlier default)
``nllb``  facebook/nllb-200-distilled-600M
``term``  terminology repair on/off

Protocol
--------
* Every fusion weight, and the cell itself, is selected on ``eval/visual_dev``.
* The selected configuration is frozen to ``config.json`` and ``eval/visual_test``
  is scored once afterwards.
* ``mt_only`` (weight 1.0) exposes translation quality alone; ``oracle_gold_en``
  is the human-English diagnostic ceiling, never a deployable result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import numpy as np

from curio.evaluate.metrics import evaluate_ranking
from curio.evaluate.stats import bootstrap_ci, paired_permutation
from curio.query import parse_query
from curio.query.bridge import CrossLingualBridge, BridgeConfig, TranslationCache, fuse_scores
from curio.represent.encoders import Encoder

METRIC_KEYS = ("ndcg@10", "ap@10", "precision@5", "recall@20", "mrr@10")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(directory: Path) -> dict:
    return {
        "queries": read_jsonl(directory / "queries.jsonl"),
        "qrels": {row["qid"]: row["grades"] for row in read_jsonl(directory / "qrels.jsonl")},
        "gallery": [str(key) for key in json.loads((directory / "gallery_ids.json").read_text(encoding="utf-8"))],
    }


def score_pool(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Encoder returned a zero query vector")
    return matrix @ (vector / norm)


def summarize(rows: list[dict]) -> dict:
    return {"metrics": {key: bootstrap_ci([row[key] for row in rows]) for key in METRIC_KEYS}, "n": len(rows)}


class SplitView:
    """Per-split state: gallery matrix, parsed queries, primary (Chinese) scores."""

    def __init__(self, split: dict, rows: np.ndarray, vectors: np.ndarray, encoder: Encoder):
        self.split = split
        self.gallery = split["gallery"]
        self.matrix = np.asarray(vectors)[rows]
        self.parsed = [parse_query(query["text_zh"]) for query in split["queries"]]
        self.primary = [
            score_pool(np.asarray(encoder.encode_texts([item.rewritten])[0], dtype=np.float32), self.matrix)
            for item in self.parsed
        ]
        self.chinese = [query["text_zh"] for query in split["queries"]]

    def rank(self, scores: np.ndarray, qid: str) -> dict:
        order = np.argsort(-scores, kind="stable")
        return evaluate_ranking([self.gallery[index] for index in order], self.split["qrels"][qid])

    def evaluate(self, channel_scores: list[np.ndarray], weight: float) -> list[dict]:
        if weight == 0.0:
            return [self.rank(primary, query["qid"])
                    for primary, query in zip(self.primary, self.split["queries"])]
        return [self.rank(fuse_scores(primary, bridge, weight), query["qid"])
                for primary, bridge, query in zip(self.primary, channel_scores, self.split["queries"])]

    def encode_channels(self, texts: list[str], encoder: Encoder) -> list[np.ndarray]:
        cache: dict[str, np.ndarray] = {}
        out = []
        for text in texts:
            if text not in cache:
                vector = np.asarray(encoder.encode_texts([text])[0], dtype=np.float32)
                cache[text] = score_pool(vector, self.matrix)
            out.append(cache[text])
        return out

    def gold_english(self, encoder: Encoder) -> list[dict]:
        rows = []
        for query in self.split["queries"]:
            vector = np.asarray(encoder.encode_texts([query["text_en"]])[0], dtype=np.float32)
            rows.append(self.rank(score_pool(vector, self.matrix), query["qid"]))
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--index", default="artifacts/siglip_adapted")
    parser.add_argument("--encoder", default="siglip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dev-dir", default="eval/visual_dev")
    parser.add_argument("--test-dir", default="eval/visual_test")
    parser.add_argument("--out", default="artifacts/bridge_2x2")
    parser.add_argument("--opus", default="Helsinki-NLP/opus-mt-zh-en")
    parser.add_argument("--nllb", default="artifacts/models/nllb-200-distilled-600M")
    parser.add_argument("--channel", default="rewritten")
    parser.add_argument("--weights", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    index_path, data_path = Path(args.index), Path(args.data)
    dev_dir, test_dir = Path(args.dev_dir), Path(args.test_dir)

    ids = json.loads((index_path / "item_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((index_path / "manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(index_path / "image_embeddings.npy", allow_pickle=False)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    if [str(item["item_id"]) for item in read_jsonl(data_path)] != ids:
        raise ValueError("Index order does not match the frozen catalogue snapshot")
    position = {key: index for index, key in enumerate(ids)}

    dev_split, test_split = load_split(dev_dir), load_split(test_dir)
    for split in (dev_split, test_split):
        missing = set(split["gallery"]) - position.keys()
        if missing:
            raise ValueError(f"Gallery IDs missing from the index; sample: {sorted(missing)[:5]}")

    encoder = Encoder(args.encoder, args.device,
                      adapter_path=manifest.get("adapter_path") if manifest.get("adapted") else None)
    encoder.encode_texts(["a porcelain vase"])
    dev = SplitView(dev_split, np.asarray([position[key] for key in dev_split["gallery"]]), vectors, encoder)
    test = SplitView(test_split, np.asarray([position[key] for key in test_split["gallery"]]), vectors, encoder)

    weights = [float(item) for item in args.weights.split(",")]
    models = [("opus", args.opus), ("nllb", args.nllb)]
    translations: dict[str, list[str]] = {}
    for label, model_path in models:
        model_cache = TranslationCache(output / "translations.jsonl", model_name=model_path)
        bridge = CrossLingualBridge(BridgeConfig(model_name=model_path), model_cache)
        translations[label] = bridge.translate(dev.chinese) + bridge.translate(test.chinese)
        print(f"{label}: translated {len(translations[label])} texts in "
              f"{bridge.translation_ms / 1000:.1f}s", flush=True)

    dev_texts: dict[tuple[str, bool], list[str]] = {}
    for label, model_path in models:
        bridge = CrossLingualBridge(BridgeConfig(model_name=model_path))
        model_translations = translations[label][:len(dev.chinese)]
        for term in (False, True):
            dev_texts[(label, term)] = [
                bridge.bridge_text(parsed, text, args.channel, term)
                for parsed, text in zip(dev.parsed, model_translations)]
    changed = {(label, term): sum(1 for source, text in zip(translations[label][:len(dev.chinese)], dev_texts[(label, term)])
                                  if source.strip().rstrip(".") not in text)
               for label, _ in models for term in (False, True)}

    curves: dict[str, dict] = {}
    for (label, term), texts in dev_texts.items():
        channel_scores = dev.encode_channels(texts, encoder)
        cell = f"{label}_{'term' if term else 'plain'}"
        curves[cell] = {
            "model": dict(models)[label], "terminology": term,
            "points": [{"weight": weight, "ndcg@10": float(np.mean([row["ndcg@10"] for row in dev.evaluate(channel_scores, weight)]))}
                       for weight in weights],
        }
        curves[cell]["best"] = max(curves[cell]["points"], key=lambda point: point["ndcg@10"])
        print(f"dev {cell}: best w={curves[cell]['best']['weight']} "
              f"nDCG={curves[cell]['best']['ndcg@10']:.4f}", flush=True)

    baseline_dev = summarize(dev.evaluate([], 0.0))
    best_cell, best_point = max(((cell, data["best"]) for cell, data in curves.items()),
                                key=lambda item: item[1]["ndcg@10"])
    selection = {
        "cell": best_cell, "weight": best_point["weight"],
        "dev_ndcg@10": best_point["ndcg@10"],
        "dev_ndcg@10_baseline": baseline_dev["metrics"]["ndcg@10"]["mean"],
        "rule": "Highest Chinese dev nDCG@10 across {model, terminology, fusion weight}; "
                "identity (weight 0) if no cell beats the baseline.",
    }
    if best_point["ndcg@10"] <= selection["dev_ndcg@10_baseline"]:
        selection.update({"cell": "identity", "weight": 0.0})

    config_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "selection": selection,
        "dev_files_sha256": {name: sha256(dev_dir / name) for name in ("queries.jsonl", "qrels.jsonl", "gallery_ids.json")},
        "test_files_sha256": {name: sha256(test_dir / name) for name in ("queries.jsonl", "qrels.jsonl", "gallery_ids.json")},
        "index_manifest_sha256": sha256(index_path / "manifest.json"),
        "models": dict(models),
    }
    (output / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    test_results: dict[str, dict] = {}
    base_rows = test.evaluate([], 0.0)
    for label, model_path in models:
        bridge = CrossLingualBridge(BridgeConfig(model_name=model_path))
        model_translations = translations[label][len(dev.chinese):]
        for term in (False, True):
            cell = f"{label}_{'term' if term else 'plain'}"
            text_list = [bridge.bridge_text(parsed, text, args.channel, term)
                         for parsed, text in zip(test.parsed, model_translations)]
            channel_scores = test.encode_channels(text_list, encoder)
            rows = test.evaluate(channel_scores, curves[cell]["best"]["weight"])
            test_results[cell] = {
                "weight": curves[cell]["best"]["weight"],
                "summary": summarize(rows),
                "delta_vs_baseline": paired_permutation([row["ndcg@10"] for row in base_rows],
                                                        [row["ndcg@10"] for row in rows]),
                "mt_only": summarize(test.evaluate(channel_scores, 1.0)),
            }

    selected_rows = test_results[selection["cell"]]["summary"] if selection["cell"] in test_results else summarize(base_rows)
    report = {
        "created_at": config_payload["created_at"],
        "git_sha": config_payload["git_sha"],
        "selection": selection,
        "dev": {
            "baseline_zh": baseline_dev,
            "oracle_gold_en": summarize(dev.gold_english(encoder)),
            "curves": curves,
            "terminology_changed_queries": {f"{label}_{'term' if term else 'plain'}": changed[(label, term)]
                                            for label, _ in models for term in (False, True)},
        },
        "test": {
            "baseline_zh": summarize(base_rows),
            "oracle_gold_en": summarize(test.gold_english(encoder)),
            "cells": test_results,
            "selected": {"cell": selection["cell"], "weight": selection["weight"], "summary": selected_rows},
        },
        "warnings": [
            "Relevance labels are disclosed AI visual-audit judgments, not independent human ground truth.",
            "The oracle English row uses the human English text of the same intent and is a diagnostic "
            "upper bound, not a deployable result.",
            "24 Chinese test queries per split: intervals are wide; a positive point estimate is not significance.",
            "Translation is offline and cached; no LLM or external API is used at query time.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"selected": selection, "test": {cell: round(data["summary"]["metrics"]["ndcg@10"]["mean"], 4)
                                                      for cell, data in test_results.items()},
                      "baseline": round(report["test"]["baseline_zh"]["metrics"]["ndcg@10"]["mean"], 4)},
                     ensure_ascii=False))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _markdown(report: dict) -> str:
    def row(name: str, summary: dict, delta: dict | None = None, weight: float | None = None) -> str:
        metric = summary["metrics"]
        ci = metric["ndcg@10"]
        change = ""
        if delta is not None:
            change = f"{delta['difference']['mean']:+.4f} (p={delta['p_value']:.3f})"
        label = f"{name} (w={weight:.1f})" if weight is not None else name
        return (f"| {label} | {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}] | "
                f"{metric['precision@5']['mean']:.4f} | {metric['ap@10']['mean']:.4f} | "
                f"{metric['recall@20']['mean']:.4f} | {metric['mrr@10']['mean']:.4f} | {change} |")

    lines = ["# Translation model x terminology repair", "",
             f"Selected on dev: `{report['selection']['cell']}` at weight "
             f"{report['selection']['weight']:.1f} (dev nDCG@10 "
             f"{report['selection']['dev_ndcg@10']:.4f} vs baseline "
             f"{report['selection']['dev_ndcg@10_baseline']:.4f}).", "",
             "| Split | Configuration | nDCG@10 (95% CI) | P@5 | AP@10 | R@20 | MRR@10 | Δ vs Chinese baseline |",
             "|---|---|---|---|---|---|---|---|",
             row("dev baseline_zh", report["dev"]["baseline_zh"]),
             row("dev oracle_gold_en (diagnostic)", report["dev"]["oracle_gold_en"]), "",
             "| Cell | Model | Terminology | Changed dev queries | Best weight (dev) | Dev nDCG@10 |",
             "|---|---|---|---|---|---|"]
    for cell, data in report["dev"]["curves"].items():
        model = Path(data["model"]).name
        lines.append(f"| {cell} | {model} | {'on' if data['terminology'] else 'off'} | "
                     f"{report['dev']['terminology_changed_queries'][cell]} | "
                     f"{data['best']['weight']:.1f} | {data['best']['ndcg@10']:.4f} |")
    lines += ["",
              "| Split | Configuration | nDCG@10 (95% CI) | P@5 | AP@10 | R@20 | MRR@10 | Δ vs Chinese baseline |",
              "|---|---|---|---|---|---|---|---|",
              row("test baseline_zh", report["test"]["baseline_zh"]),
              row("test oracle_gold_en (diagnostic)", report["test"]["oracle_gold_en"])]
    for cell, data in report["test"]["cells"].items():
        lines.append(row(f"test {cell}", data["summary"], data["delta_vs_baseline"], data["weight"]))
    lines += ["", "## Warnings", ""]
    lines += [f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
