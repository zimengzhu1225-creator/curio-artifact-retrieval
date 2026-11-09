from pathlib import Path

import numpy as np
import pytest

from curio.query import parse_query
from curio.query.bridge import (BridgeConfig, CrossLingualBridge, TranslationCache, fuse_scores,
                                needs_bridge, zscore)


def test_zscore_centers_and_scales():
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scaled = zscore(values)
    assert scaled.mean() == pytest.approx(0.0, abs=1e-6)
    assert scaled.std() == pytest.approx(1.0, abs=1e-6)


def test_zscore_handles_constant_channel():
    assert np.all(zscore(np.full(5, 0.3, dtype=np.float32)) == 0.0)


def test_fuse_weight_zero_is_identity():
    primary = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    bridge = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    assert np.allclose(fuse_scores(primary, bridge, 0.0), primary)


def test_fuse_full_weight_follows_bridge_order():
    primary = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    bridge = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    fused = fuse_scores(primary, bridge, 1.0)
    assert int(np.argmax(fused)) == 2


def test_fuse_rejects_mismatched_pools():
    with pytest.raises(ValueError):
        fuse_scores(np.zeros(3), np.zeros(4), 0.5)


def test_fuse_rejects_out_of_range_weight():
    with pytest.raises(ValueError):
        fuse_scores(np.zeros(3), np.zeros(3), 1.5)


def test_bridge_only_for_chinese():
    assert needs_bridge(parse_query("绿色窄口瓷瓶"))
    assert not needs_bridge(parse_query("a green narrow-mouth porcelain vase"))


def test_config_validation():
    assert BridgeConfig(weight=0.5, channel="rewritten").to_dict()["weight"] == 0.5
    with pytest.raises(ValueError):
        # weight outside [0, 1] must not silently pass; dataclass keeps it explicit
        fuse_scores(np.zeros(3), np.zeros(3), -0.1)


def test_translation_cache_roundtrip(tmp_path: Path):
    path = tmp_path / "cache.jsonl"
    cache = TranslationCache(path, model_name="test-model")
    cache.put("绿色的瓶子", "a green bottle")
    assert cache.get("绿色的瓶子") == "a green bottle"
    reloaded = TranslationCache(path, model_name="test-model")
    assert reloaded.get("绿色的瓶子") == "a green bottle"
    other = TranslationCache(path, model_name="another-model")
    assert other.get("绿色的瓶子") is None


def test_channel_composition_uses_parser():
    bridge = CrossLingualBridge(BridgeConfig(channel="rewritten"), TranslationCache())
    parsed = parse_query("绿色窄口瓷瓶")
    raw = bridge.bridge_text(parsed, "a green narrow-mouth vase", channel="raw")
    rewritten = bridge.bridge_text(parsed, "a green narrow-mouth vase", channel="rewritten")
    plus = bridge.bridge_text(parsed, "a green narrow-mouth vase", channel="plus_slots")
    assert raw == "a green narrow-mouth vase"
    assert rewritten and rewritten != raw
    assert raw in plus and parsed.rewritten in plus


def test_unknown_channel_rejected():
    bridge = CrossLingualBridge(BridgeConfig(channel="rewritten"), TranslationCache())
    with pytest.raises(ValueError):
        bridge.bridge_text(parse_query("青花瓷"), "blue and white porcelain", channel="mystery")
