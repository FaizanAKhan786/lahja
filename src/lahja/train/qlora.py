"""QLoRA fine-tuning: 4-bit NF4 base model + LoRA adapters (PEFT + TRL).

Usage:
    python -m lahja.train.qlora --config configs/training.yaml [--max-steps N]

Heavy imports (torch/transformers/peft/trl) happen inside main() so that
importing this module never requires a GPU stack — keeps CI light.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

log = logging.getLogger("lahja.train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps (handy for smoke runs)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg, train_cfg, lora_cfg = cfg["model"], cfg["train"], cfg["lora"]
    prepared = Path(cfg["data"].get("prepared_dir", "outputs/prepared"))

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    use_4bit = model_cfg.get("load_in_4bit", True) and torch.cuda.is_available()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if use_4bit else None

    log.info("Loading base model %s (4-bit=%s)", model_cfg["base"], use_4bit)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base"],
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    dataset = load_dataset("json", data_files={
        "train": str(prepared / "train.jsonl"),
        "eval": str(prepared / "eval.jsonl"),
    })

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    out_dir = train_cfg.get("output_dir", "outputs/adapters")
    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=train_cfg.get("epochs", 2),
        max_steps=args.max_steps or train_cfg.get("max_steps", -1),
        per_device_train_batch_size=train_cfg.get("batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("grad_accum", 8),
        learning_rate=float(train_cfg.get("lr", 2e-4)),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 200),
        bf16=torch.cuda.is_available(),
        max_length=train_cfg.get("max_seq_len", 1024),
        gradient_checkpointing=True,
        report_to=train_cfg.get("report_to", "none"),
        seed=cfg.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,
        args=sft_config,
    )

    log.info("Starting training → %s", out_dir)
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info("Adapters saved to %s", out_dir)


if __name__ == "__main__":
    main()
