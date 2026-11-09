"""Dev-selected, test-frozen evaluation of the cross-lingual query bridge.

What is measured

1. ``baseline_zh`` - the production route: rule-rewritten Chinese query against
   the frozen ``siglip_adapted`` index.
2. ``oracle_gold_en`` - the same intent written by a human in English. This is a
   *diagnostic upper bound*, not a system, and is never reported as a result.
3. ``bridge`` - offline MarianMT translation of the Chinese query, re-parsed and
   fused with the Chinese channel at a dev-selected weight.
4. ``mt_only`` - the translated channel alone, to expose translation quality as
   the bottleneck.

Weight and translated-channel form are chosen on ``eval/visual_dev`` only; the
selected configuration is frozen to disk before ``eval/visual_test`` is scored.
English queries are unchanged by the bridge and are reported separately.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from curio.evaluate.metrics import evaluate_ranking
from curio.evaluate.stats import bootstrap_ci, paired_permutation
from curio.query import parse_query
from curio.query.bridge import BridgeConfig, CrossLingualBridge, TranslationCache, fuse_scores
from curio.represent.encoders import Encoder


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


def evaluate_split(split: dict, gallery_rows: np.ndarray, matrix: np.ndarray, encoder, bridge,
                   configs: dict[str, BridgeConfig], include_oracle: bool = True) -> dict:
    """Return ``{config_name: [per-query metric dicts]}`` for Chinese queries."""
    queries = split["queries"]
    qrels = split["qrels"]
    gallery = split["gallery"]
    matrix = np.asarray(matrix)[gallery_rows]
    translations = bridge.translate([query["text_zh"] for query in queries])
    results: dict[str, list[dict]] = {name: [] for name in configs}
    diagnostics: dict[str, list[dict]] = {name: [] for name in configs}
    if include_oracle:
        results["oracle_gold_en"] = []
    for query, translation in zip(queries, translations):
        parsed = parse_query(query["text_zh"])
        primary = score_pool(np.asarray(encoder.encode_texts([parsed.rewritten])[0], dtype=np.float32), matrix)
        for name, config in configs.items():
            if config.weight == 0.0:
                fused, channel_text = primary, None
            else:
                channel_text = bridge.bridge_text(parsed, translation, config.channel)
                vector = np.asarray(encoder.encode_texts([channel_text])[0], dtype=np.float32)
                fused = fuse_scores(primary, score_pool(vector, matrix), config.weight)
            order = np.argsort(-fused, kind="stable")
            ranking = [gallery[index] for index in order]
            results[name].append(evaluate_ranking(ranking, qrels[query["qid"]]))
            diagnostics[name].append({"qid": query["qid"], "translation": translation,
                                      "translated_channel": channel_text,
                                      "top1": ranking[0]})
        if include_oracle:
            vector = np.asarray(encoder.encode_texts([query["text_en"]])[0], dtype=np.float32)
            order = np.argsort(-score_pool(vector, matrix), kind="stable")
            results["oracle_gold_en"].append(
                evaluate_ranking([gallery[index] for index in order], qrels[query["qid"]]))
    return {"metrics": results, "diagnostics": diagnostics}


def summarize(rows: list[dict]) -> dict:
    keys = ("ndcg@10", "ap@10", "precision@5", "recall@20", "mrr@10")
    return {"metrics": {key: bootstrap_ci([row[key] for row in rows]) for key in keys}, "n": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--index", default="artifacts/siglip_adapted")
    parser.add_argument("--encoder", default="siglip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dev-dir", default="eval/visual_dev")
    parser.add_argument("--test-dir", default="eval/visual_test")
    parser.add_argument("--out", default="artifacts/bridge_eval")
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-zh-en")
    parser.add_argument("--channels", default="raw,rewritten,plus_slots")
    parser.add_argument("--weights", default="0.2,0.3,0.4,0.5,0.6,0.7")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    index_path, data_path = Path(args.index), Path(args.data)
    dev_dir, test_dir = Path(args.dev_dir), Path(args.test_dir)

    ids = json.loads((index_path / "item_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((index_path / "manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(index_path / "image_embeddings.npy", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("Invalid image index")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate item IDs in the index")
    if [str(item["item_id"]) for item in read_jsonl(data_path)] != ids:
        raise ValueError("Index order does not match the frozen catalogue snapshot")
    position = {key: index for index, key in enumerate(ids)}

    dev, test = load_split(dev_dir), load_split(test_dir)
    for split in (dev, test):
        split["rows"] = np.asarray([position[key] for key in split["gallery"]], dtype=np.int64)
        unknown = set(split["gallery"]) - position.keys()
        if unknown:
            raise ValueError(f"Gallery IDs missing from the index; sample: {sorted(unknown)[:5]}")

    encoder = Encoder(args.encoder, args.device,
                      adapter_path=manifest.get("adapter_path") if manifest.get("adapted") else None)
    encoder.encode_texts(["a porcelain vase"])

    cache = TranslationCache(output / "translations.jsonl", model_name=args.model)
    channels = [item.strip() for item in args.channels.split(",")]
    weights = [float(item) for item in args.weights.split(",")]
    grid = {"baseline_zh": BridgeConfig(model_name=args.model, weight=0.0)}
    for channel in channels:
        for weight in weights:
            grid[f"bridge_{channel}_w{weight}"] = BridgeConfig(model_name=args.model, weight=weight, channel=channel)
    grid["mt_only"] = BridgeConfig(model_name=args.model, weight=1.0, channel="rewritten")

    bridge = CrossLingualBridge(next(iter(grid.values())), cache)
    started = time.perf_counter()
    dev_run = evaluate_split(dev, dev["rows"], vectors, encoder, bridge, grid)
    print(f"dev grid: {len(grid)} configs x {len(dev['queries'])} queries in {time.perf_counter() - started:.1f}s", flush=True)

    dev_summary = {name: summarize(rows) for name, rows in dev_run["metrics"].items()}
    baseline_dev = dev_summary["baseline_zh"]["metrics"]["ndcg@10"]["mean"]
    candidates = sorted(((summary["metrics"]["ndcg@10"]["mean"], name) for name, summary in dev_summary.items()
                         if name.startswith("bridge_")), reverse=True)
    best_score, selected_name = candidates[0]
    selected = grid[selected_name]
    if best_score <= baseline_dev:
        selected, selected_name = BridgeConfig(model_name=args.model, weight=0.0), "baseline_zh"

    config_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected": selected.to_dict(), "selected_name": selected_name,
        "dev_ndcg@10_baseline_zh": baseline_dev,
        "dev_ndcg@10_selected": dev_summary[selected_name]["metrics"]["ndcg@10"]["mean"],
        "dev_ndcg@10_oracle_gold_en": dev_summary["oracle_gold_en"]["metrics"]["ndcg@10"]["mean"],
        "dev_files_sha256": {name: sha256(dev_dir / name) for name in ("queries.jsonl", "qrels.jsonl", "gallery_ids.json")},
        "test_files_sha256": {name: sha256(test_dir / name) for name in ("queries.jsonl", "qrels.jsonl", "gallery_ids.json")},
        "index_manifest_sha256": sha256(index_path / "manifest.json"),
        "data_sha256": sha256(data_path),
        "selection_rule": "Highest Chinese dev nDCG@10 across translated-channel forms and fusion weights; "
                          "identity (weight 0) if nothing beats the baseline.",
    }
    (output / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    test_configs = {"baseline_zh": grid["baseline_zh"], "mt_only": grid["mt_only"], selected_name: selected} \
        if selected_name != "baseline_zh" else {"baseline_zh": grid["baseline_zh"], "mt_only": grid["mt_only"]}
    test_run = evaluate_split(test, test["rows"], vectors, encoder, bridge, test_configs)
    test_summary = {name: summarize(rows) for name, rows in test_run["metrics"].items()}
    pairings = {}
    base_rows = test_run["metrics"]["baseline_zh"]
    for name, rows in test_run["metrics"].items():
        if name == "baseline_zh":
            continue
        pairings[name] = paired_permutation([row["ndcg@10"] for row in base_rows],
                                            [row["ndcg@10"] for row in rows])

    english = []
    for query in test["queries"]:
        vector = np.asarray(encoder.encode_texts([query["text_en"]])[0], dtype=np.float32)
        order = np.argsort(-score_pool(vector, vectors[test["rows"]]), kind="stable")
        english.append(evaluate_ranking([test["gallery"][index] for index in order], test["qrels"][query["qid"]]))

    report = {
        "created_at": config_payload["created_at"],
        "git_sha": _git_sha(),
        "translator": {"model": args.model, "channels": channels, "weights": weights,
                       "translation_ms": round(bridge.translation_ms, 1),
                       "cached_entries": len(cache.entries)},
        "dev": {"baseline_zh": dev_summary["baseline_zh"], "selected_name": selected_name,
                "selected": dev_summary[selected_name], "oracle_gold_en": dev_summary["oracle_gold_en"],
                "grid": [{"name": name, "ndcg@10": summary["metrics"]["ndcg@10"]["mean"],
                          "precision@5": summary["metrics"]["precision@5"]["mean"]}
                         for name, summary in sorted(dev_summary.items(),
                                                     key=lambda item: -item[1]["metrics"]["ndcg@10"]["mean"])]},
        "test": {"baseline_zh": test_summary["baseline_zh"], "oracle_gold_en": test_summary["oracle_gold_en"],
                 "english_reference": summarize(english),
                 "variants": {name: {"summary": test_summary[name], "delta_vs_baseline": pairings[name]}
                              for name in test_configs if name != "baseline_zh"}},
        "warnings": [
            "Relevance labels are disclosed AI visual-audit judgments, not independent human ground truth.",
            "The oracle English row uses the human English text of the same intent and is a diagnostic "
            "upper bound, not a deployable result.",
            "24 Chinese test queries: intervals are wide and a positive point estimate is not significance.",
            "Translation is offline and cached; no LLM or external API is used at query time.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "diagnostics.jsonl").write_text(
        "".join(json.dumps({"split": "test", **row}, ensure_ascii=False) + "\n"
                for row in test_run["diagnostics"].get(selected_name, [])), encoding="utf-8")
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"selected": selected_name,
                      "dev_ndcg": report["dev"]["selected"]["metrics"]["ndcg@10"]["mean"],
                      "test_zh_baseline": report["test"]["baseline_zh"]["metrics"]["ndcg@10"]["mean"],
                      "test_zh_selected": report["test"]["variants"].get(selected_name, {}).get("summary", {}).get("metrics", {}).get("ndcg@10", {}).get("mean"),
                      "oracle_en": report["test"]["oracle_gold_en"]["metrics"]["ndcg@10"]["mean"]},
                     ensure_ascii=False))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _markdown(report: dict) -> str:
    def column(summary):
        metric = summary["metrics"]["ndcg@10"]
        return (f"{metric['mean']:.4f} [{metric['lower']:.4f}, {metric['upper']:.4f}]", summary["metrics"])

    lines = ["# Cross-lingual query bridge", "",
             f"Translator: `{report['translator']['model']}` (offline, cached, "
             f"{report['translator']['cached_entries']} entries).", "",
             "| Split | Configuration | nDCG@10 (95% CI) | P@5 | AP@10 | R@20 | MRR@10 | Δ vs Chinese baseline |",
             "|---|---|---|---|---|---|---|---|"]
    for split in ("dev", "test"):
        block = report[split]
        rows = [("baseline_zh", block["baseline_zh"])]
        if split == "dev":
            rows.append((block["selected_name"], block["selected"]))
        else:
            rows.extend((name, payload["summary"]) for name, payload in block["variants"].items())
        rows.append(("oracle_gold_en (diagnostic)", block["oracle_gold_en"]))
        for name, summary in rows:
            n, metrics = column(summary)
            delta = ""
            if split == "test" and name in block["variants"]:
                change = block["variants"][name]["delta_vs_baseline"]
                delta = f"{change['difference']['mean']:+.4f} (p={change['p_value']:.3f})"
            lines.append(f"| {split} | {name} | {n} | {metrics['precision@5']['mean']:.4f} | "
                         f"{metrics['ap@10']['mean']:.4f} | {metrics['recall@20']['mean']:.4f} | "
                         f"{metrics['mrr@10']['mean']:.4f} | {delta} |")
    english = report["test"]["english_reference"]["metrics"]
    lines += ["", f"English reference on the same test queries: nDCG@10 {english['ndcg@10']['mean']:.4f}, "
                  f"P@5 {english['precision@5']['mean']:.4f}.", "", "## Warnings", ""]
    lines += [f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
