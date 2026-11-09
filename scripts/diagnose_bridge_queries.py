"""Per-query diagnostics for the selected bridge cell (not part of the pipeline)."""

import json
from pathlib import Path

import numpy as np

from curio.evaluate.metrics import evaluate_ranking
from curio.query import parse_query
from curio.query.bridge import BridgeConfig, CrossLingualBridge, TranslationCache, fuse_scores
from curio.query.terminology import repair_translation
from curio.represent.encoders import Encoder

INDEX = Path("artifacts/siglip_adapted")
MODEL = "artifacts/models/nllb-200-distilled-600M"
SPLIT = Path("eval/visual_test")

ids = json.loads((INDEX / "item_ids.json").read_text(encoding="utf-8"))
manifest = json.loads((INDEX / "manifest.json").read_text(encoding="utf-8"))
vectors = np.load(INDEX / "image_embeddings.npy")
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
position = {key: index for index, key in enumerate(ids)}
queries = [json.loads(line) for line in (SPLIT / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
qrels = {row["qid"]: row["grades"]
         for row in (json.loads(line) for line in (SPLIT / "qrels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
gallery = [str(key) for key in json.loads((SPLIT / "gallery_ids.json").read_text(encoding="utf-8"))]
matrix = vectors[[position[key] for key in gallery]]

encoder = Encoder("siglip", "auto", adapter_path=manifest.get("adapter_path"))
bridge = CrossLingualBridge(BridgeConfig(model_name=MODEL),
                            TranslationCache("artifacts/bridge_2x2/translations.jsonl", model_name=MODEL))
translations = bridge.translate([query["text_zh"] for query in queries])


def dense(text: str) -> np.ndarray:
    vector = np.asarray(encoder.encode_texts([text])[0], dtype=np.float32)
    return matrix @ (vector / np.linalg.norm(vector))


def ndcg(scores: np.ndarray, qid: str) -> float:
    order = np.argsort(-scores, kind="stable")
    return float(evaluate_ranking([gallery[index] for index in order], qrels[qid])["ndcg@10"])


rows = []
for query, translation in zip(queries, translations):
    parsed = parse_query(query["text_zh"])
    primary = dense(parsed.rewritten)
    plain = dense(bridge.bridge_text(parsed, translation, "rewritten", False))
    repaired_text = bridge.bridge_text(parsed, translation, "rewritten", True)
    repaired = dense(repaired_text)
    rows.append((query["qid"], query["text_zh"], ndcg(primary, query["qid"]),
                 ndcg(fuse_scores(primary, plain, 0.4), query["qid"]),
                 ndcg(fuse_scores(primary, repaired, 0.4), query["qid"]),
                 repair_translation(query["text_zh"], translation, parsed.slots) != translation))

rows.sort(key=lambda row: -(row[4] - row[2]))
print(f"{'qid':6} {'base':>6} {'nllb':>6} {'term':>6} {'delta':>7}  rep  query")
for row in rows:
    print(f"{row[0]:6} {row[2]:6.4f} {row[3]:6.4f} {row[4]:6.4f} {row[4] - row[2]:+7.4f}  "
          f"{'Y' if row[5] else '-':3}  {row[1][:32]}")
print()
print("mean base/nllb/term: %.4f %.4f %.4f" % tuple(np.mean([row[i] for row in rows]) for i in (2, 3, 4)))
print("wins/losses:", sum(1 for row in rows if row[4] > row[2] + 1e-9),
      sum(1 for row in rows if row[4] < row[2] - 1e-9))
