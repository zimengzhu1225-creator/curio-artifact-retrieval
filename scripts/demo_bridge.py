"""Show one Chinese query before and after the cross-lingual bridge.

Usage::

    python scripts/demo_bridge.py --query 绿色窄口瓷瓶
    python scripts/demo_bridge.py --query 蓝色龙纹盘子 --top 5
    python scripts/demo_bridge.py --query 找带竖棱的盖盒 --compare-terminology

The script reuses the frozen index, the frozen parser, and the dev-selected
configuration from ``artifacts/bridge_2x2/config.json`` (or the older
``artifacts/bridge_eval/config.json``); it never writes to the catalogue or the
index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from curio.query import parse_query
from curio.query.bridge import BridgeConfig, CrossLingualBridge, TranslationCache, fuse_scores, needs_bridge
from curio.represent.encoders import Encoder


def load_selection(path: Path) -> dict:
    """Return {model_name, weight, channel, terminology} from either config format."""
    if not path.exists():
        return {"model_name": "Helsinki-NLP/opus-mt-zh-en", "weight": 0.0,
                "channel": "rewritten", "terminology": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "selection" in payload:
        selection = payload["selection"]
        cell = str(selection.get("cell", "identity"))
        label = cell.split("_")[0]
        models = payload.get("models", {})
        return {"model_name": models.get(label, "Helsinki-NLP/opus-mt-zh-en"),
                "weight": float(selection.get("weight", 0.0)),
                "channel": "rewritten",
                "terminology": cell.endswith("_term")}
    selected = payload.get("selected", {})
    return {"model_name": selected.get("model_name", "Helsinki-NLP/opus-mt-zh-en"),
            "weight": float(selected.get("weight", 0.0)),
            "channel": selected.get("channel", "rewritten"),
            "terminology": bool(selected.get("terminology", False))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--index", default="artifacts/siglip_adapted")
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--config", default="artifacts/bridge_2x2/config.json")
    parser.add_argument("--cache", default="artifacts/bridge_2x2/translations.jsonl")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compare-terminology", action="store_true",
                        help="also show the same channel without terminology repair")
    args = parser.parse_args()

    selection = load_selection(Path(args.config))
    weight = selection["weight"]
    channel = selection["channel"]
    model_name = selection["model_name"]

    index = Path(args.index)
    ids = json.loads((index / "item_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(index / "image_embeddings.npy", allow_pickle=False)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    items = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(item["item_id"]): item for item in items}

    encoder = Encoder("siglip", args.device,
                      adapter_path=manifest.get("adapter_path") if manifest.get("adapted") else None)
    parsed = parse_query(args.query)

    def dense(text: str) -> np.ndarray:
        vector = np.asarray(encoder.encode_texts([text])[0], dtype=np.float32)
        return vectors @ (vector / np.linalg.norm(vector))

    primary = dense(parsed.rewritten)
    rows = [("未桥接（中文单通道）", primary)]
    if weight > 0 and needs_bridge(parsed):
        bridge = CrossLingualBridge(
            BridgeConfig(model_name=model_name, weight=weight, channel=channel,
                         terminology=selection["terminology"]),
            TranslationCache(args.cache, model_name=model_name))
        translation = bridge.translate([args.query])[0]
        print(f"翻译模型: {Path(model_name).name}")
        print(f"MT 译文 : {translation}")
        for term in ([False, True] if args.compare_terminology else [selection["terminology"]]):
            channel_text = bridge.bridge_text(parsed, translation, channel, term)
            if term:
                print(f"术语纠错: {channel_text}")
            label = f"跨语言桥接 w={weight}" + ("（+术语纠错）" if term else "（无术语纠错）")
            rows.append((label, fuse_scores(primary, dense(channel_text), weight)))

    for label, scores in rows:
        order = np.argsort(-scores, kind="stable")[:args.top]
        print(f"\n== {label} ==")
        for rank, position in enumerate(order, 1):
            item = by_id[ids[position]]
            print(f"{rank}. {float(scores[position]):.4f}  {item.get('title', '')[:56]}  "
                  f"[{item.get('source')}]")
    if len(rows) == 1:
        print("\n（未启用桥接：该查询不是中文，或配置中权重为 0）")


if __name__ == "__main__":
    main()
