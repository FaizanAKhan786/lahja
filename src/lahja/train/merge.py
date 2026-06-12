"""Merge trained LoRA adapters into a standalone model for serving/quantization.

Usage:
    python -m lahja.train.merge --config configs/training.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

log = logging.getLogger("lahja.merge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    adapter_dir = cfg["train"].get("output_dir", "outputs/adapters")
    merged_dir = cfg["train"].get("merged_dir", "outputs/merged")

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    log.info("Loading adapters from %s", adapter_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(merged_dir)
    log.info("Merged model saved to %s", merged_dir)


if __name__ == "__main__":
    main()
