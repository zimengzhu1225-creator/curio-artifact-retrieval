"""Dev-selected, test-frozen evaluation of the contradiction-gated reranker.

Protocol (mirrors the repository's frozen-visual discipline):

1. Dense recall on the production index (``artifacts/siglip_adapted``) with the
   rule-rewritten query, identical to the default ``image_only`` route.
2. Every reranker hyper-parameter is chosen on ``eval/visual_dev`` only.
3. The selected configuration is written to disk, then applied once to
   ``eval/visual_test`` together with fixed ablations.
4. Nothing here writes to frozen data, indexes, probes, or the production
   selection file; outputs go to a fresh ``--out`` directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from curio.evaluate.metrics import evaluate_ranking
from curio.evaluate.stats import bootstrap_ci, paired_permutation
from curio.query import parse_query
from curio.represent.encoders import Encoder
from curio.represent.probes import LinearProbes
from curio.retrieve.rerank import RerankConfig, rerank_ranking


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_index(index: Path):
    ids = json.loads((index / "item_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(index / "image_embeddings.npy", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("Invalid image index")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Index contains zero vectors")
    return ids, vectors / norms, manifest


def load_probes(path: Path, index_sha: str, data_sha: str, dimension: int) -> dict:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if report.get("index_sha256") != index_sha:
        raise ValueError("Probe directory was not trained on this index; refusing to mix feature spaces")
    if data_sha and report.get("data_sha256") != data_sha:
        raise ValueError("Probe directory data hash differs from the frozen snapshot")
    if int(report.get("feature_dimension", -1)) != dimension:
        raise ValueError("Probe feature dimension differs from the index")
    return {"report": report, "predictor": LinearProbes(path)}


def run_queries(items_by_id, ids, vectors, predictor, encoder, queries, gallery_rows,
                qrels, languages, configs):
    """Evaluate every configuration over one query set; returns per-config metric rows."""
    records = {name: [] for name in configs}
    diagnostics = {name: [] for name in configs}
    for query in queries:
        for language in languages:
            parsed = parse_query(query[f"text_{language}"])
            text_vector = np.asarray(encoder.encode_texts([parsed.rewritten]), dtype=np.float32)[0]
            pool_vectors = vectors[gallery_rows]
            dense = pool_vectors @ (text_vector / np.linalg.norm(text_vector))
            for name, config in configs.items():
                order, diag = rerank_ranking(dense, parsed, predictor, gallery_rows, config)
                ranking = [ids[row] for row in order]
                scores = evaluate_ranking(ranking, qrels[query["qid"]])
                records[name].append({"qid": query["qid"], "language": language, **scores})
                diagnostics[name].append({"qid": query["qid"], "language": language,
                                          "slots": diag.get("covered_slots", []), **{k: v for k, v in diag.items() if k != "covered_slots"}})
    return records, diagnostics


def summarize(rows):
    metrics = {key: bootstrap_ci([row[key] for row in rows])
               for key in ("ndcg@10", "ap@10", "precision@5", "recall@20", "mrr@10")}
    per_language = {}
    for language in sorted({row["language"] for row in rows}):
        subset = [row for row in rows if row["language"] == language]
        per_language[language] = {key: bootstrap_ci([row[key] for row in subset])["mean"]
                                  for key in ("ndcg@10", "precision@5")}
    return {"metrics": metrics, "per_language": per_language, "n": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--index", default="artifacts/siglip_adapted")
    parser.add_argument("--probes", default="artifacts/probes/vision_metadata_adapted")
    parser.add_argument("--encoder", default="siglip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dev-dir", default="eval/visual_dev")
    parser.add_argument("--test-dir", default="eval/visual_test")
    parser.add_argument("--languages", default="en,zh")
    parser.add_argument("--out", default="artifacts/rerank_eval")
    args = parser.parse_args()

    languages = [item.strip() for item in args.languages.split(",")]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    data_path, index_path, probes_path = Path(args.data), Path(args.index), Path(args.probes)
    dev_dir, test_dir = Path(args.dev_dir), Path(args.test_dir)

    data_sha = sha256(data_path)
    items = read_jsonl(data_path)
    ids, vectors, manifest = load_index(index_path)
    if [str(item["item_id"]) for item in items] and len(ids) != len(items):
        raise ValueError("Index does not cover the frozen catalog")
    by_id = {str(item["item_id"]): item for item in items}
    position = {key: idx for idx, key in enumerate(ids)}
    probe_bundle = load_probes(probes_path, manifest.get("index_sha256", ""), data_sha, vectors.shape[1])
    predictions = probe_bundle["predictor"].predict(vectors)

    encoder = Encoder(args.encoder, args.device, adapter_path=manifest.get("adapter_path") if manifest.get("adapted") else None)
    encoder.encode_texts(["a porcelain vase"])

    dev = {
        "queries": read_jsonl(dev_dir / "queries.jsonl"),
        "qrels": {row["qid"]: row["grades"] for row in read_jsonl(dev_dir / "qrels.jsonl")},
        "gallery": [str(key) for key in json.loads((dev_dir / "gallery_ids.json").read_text(encoding="utf-8"))],
    }
    test = {
        "queries": read_jsonl(test_dir / "queries.jsonl"),
        "qrels": {row["qid"]: row["grades"] for row in read_jsonl(test_dir / "qrels.jsonl")},
        "gallery": [str(key) for key in json.loads((test_dir / "gallery_ids.json").read_text(encoding="utf-8"))],
    }
    for split in (dev, test):
        split["rows"] = np.asarray([position[key] for key in split["gallery"]], dtype=np.int64)

    grid: dict[str, RerankConfig] = {"baseline_dense": RerankConfig(mode="none")}
    for weight in (0.02, 0.05, 0.1, 0.15, 0.3):
        grid[f"naive_w{weight}"] = RerankConfig(mode="naive", support_weight=weight, aggregation="mean")
    for top_n in (10, 20, 30, 60):
        for aggregation in ("mean", "min"):
            for support_weight in (0.0, 0.05, 0.1, 0.2):
                for conflict_weight in (0.0, 0.1, 0.3):
                    for margin in (0.0, 0.2):
                        if support_weight == 0.0 and conflict_weight == 0.0:
                            continue
                        name = f"gate_n{top_n}_{aggregation}_s{support_weight}_c{conflict_weight}_m{margin}"
                        grid[name] = RerankConfig(mode="gate", top_n=top_n, aggregation=aggregation,
                                                  support_weight=support_weight, conflict_weight=conflict_weight,
                                                  conflict_margin=margin)

    started = time.perf_counter()
    dev_records, dev_diag = run_queries(by_id, ids, vectors, predictions, encoder, dev["queries"],
                                        dev["rows"], dev["qrels"], languages, grid)
    print(f"dev grid: {len(grid)} configs, {len(dev['queries']) * len(languages)} runs, {time.perf_counter() - started:.1f}s", flush=True)

    dev_summary = {name: summarize(rows) for name, rows in dev_records.items()}
    baseline_dev = dev_summary["baseline_dense"]["metrics"]["ndcg@10"]["mean"]
    ordered = sorted(dev_summary.items(),
                     key=lambda kv: (-kv[1]["metrics"]["ndcg@10"]["mean"],
                                     sum(v for v in (grid[kv[0]].support_weight, grid[kv[0]].conflict_weight) if v),
                                     kv[0]))
    selected_name = next(name for name, _ in ordered if name != "baseline_dense")
    selected = grid[selected_name]
    if dev_summary[selected_name]["metrics"]["ndcg@10"]["mean"] <= baseline_dev:
        selected, selected_name = RerankConfig(mode="none"), "baseline_dense"

    naive_best_name = max((name for name in grid if name.startswith("naive_")),
                          key=lambda name: dev_summary[name]["metrics"]["ndcg@10"]["mean"])
    support_only = RerankConfig(mode="gate", top_n=selected.top_n, aggregation=selected.aggregation,
                                support_weight=selected.support_weight, conflict_weight=0.0)
    conflict_only = RerankConfig(mode="gate", top_n=selected.top_n, aggregation=selected.aggregation,
                                 support_weight=0.0, conflict_weight=selected.conflict_weight or 0.3,
                                 conflict_margin=selected.conflict_margin)
    test_configs = {
        "baseline_dense": RerankConfig(mode="none"),
        "naive_best_dev": grid[naive_best_name],
        "gate_support_only": support_only,
        "gate_conflict_only": conflict_only,
        "gate_full_selected": selected,
    }
    test_records, test_diag = run_queries(by_id, ids, vectors, predictions, encoder, test["queries"],
                                          test["rows"], test["qrels"], languages, test_configs)
    test_summary = {name: summarize(rows) for name, rows in test_records.items()}

    def paired(name):
        base = {(row["qid"], row["language"]): row["ndcg@10"] for row in test_records["baseline_dense"]}
        other = {(row["qid"], row["language"]): row["ndcg@10"] for row in test_records[name]}
        keys = sorted(base)
        return paired_permutation([base[key] for key in keys], [other[key] for key in keys])

    coverage = {}
    for split_name, split in (("dev", dev), ("test", test)):
        covered = 0
        slot_counts = []
        for query in split["queries"]:
            for language in languages:
                parsed = parse_query(query[f"text_{language}"])
                slots = [task for task, values in parsed.slots.items()
                         if values and task in predictions]
                slot_counts.append(len(slots))
                covered += bool(slots)
        coverage[split_name] = {
            "runs": len(split["queries"]) * len(languages),
            "runs_with_covered_slots": covered,
            "share_with_covered_slots": round(covered / max(len(slot_counts), 1), 4),
            "mean_covered_slots": round(float(np.mean(slot_counts)) if slot_counts else 0.0, 3),
        }

    config_path = output / "config.json"
    config_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected": selected.to_dict(),
        "selected_name": selected_name,
        "dev_ndcg@10_baseline": baseline_dev,
        "dev_ndcg@10_selected": dev_summary[selected_name]["metrics"]["ndcg@10"]["mean"],
        "dev_files_sha256": {name: sha256(dev_dir / name) for name in ("queries.jsonl", "qrels.jsonl", "gallery_ids.json")},
        "test_files_sha256": {name: sha256(test_dir / name) for name in ("queries.jsonl", "qrels.jsonl", "gallery_ids.json")},
        "data_sha256": data_sha,
        "index_manifest_sha256": sha256(index_path / "manifest.json"),
        "probes_report_sha256": sha256(probes_path / "report.json"),
        "selection_rule": "Highest bilingual dev nDCG@10 mean; falls back to dense when no configuration beats the baseline.",
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "created_at": config_payload["created_at"],
        "git_sha": _git_sha(),
        "config": {name: grid[name].to_dict() for name in sorted(grid)},
        "dev": {"baseline": dev_summary["baseline_dense"],
                "selected": dev_summary[selected_name], "selected_name": selected_name,
                "top_10": [{"name": name, "config": grid[name].to_dict(),
                            "ndcg@10": summary["metrics"]["ndcg@10"]["mean"]} for name, summary in ordered[:10]],
                "naive_best_name": naive_best_name, "naive_best": dev_summary[naive_best_name]},
        "test": {name: {"summary": test_summary[name], "delta_vs_baseline": paired(name)} for name in test_configs},
        "coverage": coverage,
        "warnings": [
            "Relevance labels are disclosed AI visual-audit judgments, not independent human ground truth.",
            "The dev grid contains many correlated trial configurations on 48 runs; test differences within the reported interval are not claimed as significant.",
            "Probe outputs are weak-supervision predictions from museum-metadata labels; they do not verify authenticity, era, or material.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "dev_grid.jsonl").write_text(
        "".join(json.dumps({"name": name, "config": grid[name].to_dict(),
                            "ndcg@10": summary["metrics"]["ndcg@10"]["mean"],
                            "precision@5": summary["metrics"]["precision@5"]["mean"]}, ensure_ascii=False) + "\n"
                for name, summary in ordered), encoding="utf-8")
    (output / "report.md").write_text(_markdown(report, selected_name), encoding="utf-8")
    print(json.dumps({"selected": selected_name, "dev_ndcg": report["dev"]["selected"]["metrics"]["ndcg@10"]["mean"],
                      "test_delta": report["test"]["gate_full_selected"]["delta_vs_baseline"]["difference"]["mean"]},
                     ensure_ascii=False))


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _markdown(report, selected_name) -> str:
    lines = ["# Contradiction-gated reranker", "",
             f"Selected on dev: **{selected_name}**", "",
             "| Split | Configuration | nDCG@10 (95% CI) | P@5 | AP@10 | R@20 | MRR@10 | ΔnDCG@10 vs dense |",
             "|---|---|---|---|---|---|---|---|"]
    for split in ("dev", "test"):
        block = report[split]
        rows = ([(block["selected_name"], block["selected"])]
                if split == "dev" else [(name, block[name]["summary"]) for name in block])
        for name, summary in rows:
            m = summary["metrics"]
            delta = "" if split == "dev" else f"{block[name]['delta_vs_baseline']['difference']['mean']:+.4f} (p={block[name]['delta_vs_baseline']['p_value']:.3f})"
            lines.append(
                f"| {split} | {name} | {m['ndcg@10']['mean']:.4f} [{m['ndcg@10']['lower']:.4f}, {m['ndcg@10']['upper']:.4f}] | "
                f"{m['precision@5']['mean']:.4f} | {m['ap@10']['mean']:.4f} | {m['recall@20']['mean']:.4f} | "
                f"{m['mrr@10']['mean']:.4f} | {delta} |")
    lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
