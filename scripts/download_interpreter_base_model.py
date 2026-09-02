#!/usr/bin/env python3
"""Download the exact pinned upstream Qwen snapshot used for QLoRA training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_lab.interpreter_finetuning import (
    DEFAULT_CONFIG,
    FineTuningError,
    load_qlora_config,
    verify_base_snapshot,
    write_snapshot_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and checksum the frozen Qwen safetensors/tokenizer snapshot."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install requirements-training.txt before downloading the training model.", file=sys.stderr)
        return 2

    try:
        config = load_qlora_config(args.config)
        base = config["base_model"]
        if args.output.exists() and any(args.output.iterdir()):
            raise FineTuningError(
                "The download directory must be empty; never mix a frozen snapshot with existing files."
            )
        args.output.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=base["repository"],
            revision=base["revision"],
            local_dir=args.output,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "model-*.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template*.jinja",
            ],
        )
        write_snapshot_receipt(
            args.output, repository=base["repository"], revision=base["revision"]
        )
        identity = verify_base_snapshot(args.output, config)
        print(json.dumps(identity, indent=2, ensure_ascii=False))
        print(f"Verified frozen snapshot at {args.output.resolve()}")
        return 0
    except Exception as exc:
        print(f"Frozen model download failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
