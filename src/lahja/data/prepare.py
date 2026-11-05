"""Prepare raw instruction JSONL into formatted train/eval splits.

Usage:
    python -m lahja.data.prepare --config configs/training.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import yaml

from lahja.data.format import load_jsonl, prepare_examples, to_chat_messages

log = logging.getLogger("lahja.prepare")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg["data"]
    random.seed(cfg.get("seed", 42))

    examples = prepare_examples(
        load_jsonl(data_cfg["train_file"]),
        require_arabic=data_cfg.get("require_arabic", True),
    )
    log.info("Loaded %d examples after filtering/dedup", len(examples))

    random.shuffle(examples)
    eval_frac = data_cfg.get("eval_fraction", 0.05)
    n_eval = max(1, int(len(examples) * eval_frac))
    eval_set, train_set = examples[:n_eval], examples[n_eval:]

    out_dir = Path(data_cfg.get("prepared_dir", "outputs/prepared"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split in (("train", train_set), ("eval", eval_set)):
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps(
                    {"messages": to_chat_messages(ex), "dialect": ex.dialect},
                    ensure_ascii=False,
                ) + "\n"
                for ex in split
            )
        log.info("Wrote %d examples → %s", len(split), path)


if __name__ == "__main__":
    main()
