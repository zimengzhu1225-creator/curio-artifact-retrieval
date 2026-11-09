"""Download a pinned translation model from ModelScope into artifacts/models.

Used because huggingface.co is unreachable from this network. The file list is
fixed below so the snapshot is reproducible; no wildcard or latest-revision
lookup is performed at download time.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

FILES = {
    "facebook/nllb-200-distilled-600M": [
        ("config.json", 846),
        ("generation_config.json", 189),
        ("pytorch_model.bin", 2460457927),
        ("sentencepiece.bpe.model", 4852054),
        ("special_tokens_map.json", 3548),
        ("tokenizer.json", 17331176),
        ("tokenizer_config.json", 564),
    ],
}

BASE = "https://modelscope.cn/models/{repo}/resolve/master/{name}"


def download(repo: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name, size in FILES[repo]:
        destination = target / name
        if destination.exists() and destination.stat().st_size == size:
            print(f"[skip] {name} ({size} bytes)", flush=True)
            continue
        url = BASE.format(repo=repo, name=name)
        started = time.perf_counter()
        request = urllib.request.Request(url, headers={"User-Agent": "curio-download"})
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
            total = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
                if size > 50_000_000 and total % (200 << 20) < (1 << 20):
                    print(f"  {name}: {total / 1e6:.0f} MB / {size / 1e6:.0f} MB", flush=True)
        elapsed = time.perf_counter() - started
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        print(f"[ok] {name} {total} bytes in {elapsed:.1f}s sha256={digest[:16]}", flush=True)
        if total != size:
            raise SystemExit(f"Size mismatch for {name}: {total} != {size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--out", default="artifacts/models/nllb-200-distilled-600M")
    args = parser.parse_args()
    download(args.repo, Path(args.out))
    print("done", flush=True)
    sys.exit(0)
