"""Evaluation harness: dialect ID accuracy/F1, perplexity, optional LLM-as-judge.

Usage:
    python -m lahja.eval.evaluate --config configs/training.yaml

Writes JSON + Markdown reports to `reports/`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

log = logging.getLogger("lahja.eval")


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Dependency-free macro-F1 over string labels."""
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    import torch

    losses, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            losses += out.loss.item()
            count += 1
    return math.exp(losses / max(count, 1))


def dialect_id_eval(model, tokenizer, rows: list[dict], dialects: list[str]) -> dict:
    """Zero-shot dialect classification: ask the model to label each text."""
    import torch

    prompt_tmpl = (
        "صنّف لهجة النص التالي. اختر واحدة فقط من: {labels}.\n"
        "النص: {text}\n"
        "اللهجة:"
    )
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for row in rows:
            text = row["messages"][1]["content"]
            prompt = prompt_tmpl.format(labels="، ".join(dialects), text=text[:400])
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
            pred = next((d for d in dialects if d in answer), dialects[0])
            y_true.append(row.get("dialect", "gulf"))
            y_pred.append(pred)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "support": dict(Counter(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-dialect-id", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    prepared = Path(cfg["data"].get("prepared_dir", "outputs/prepared"))
    merged_dir = cfg["train"].get("merged_dir", "outputs/merged")
    dialects = cfg.get("eval", {}).get("dialects", ["قطري", "إماراتي", "سعودي", "كويتي"])

    with (prepared / "eval.jsonl").open(encoding="utf-8") as eval_file:
        rows = [json.loads(line) for line in eval_file]
    texts = [r["messages"][2]["content"] for r in rows]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model %s", merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    report: dict = {
        "model": merged_dir,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_eval": len(rows),
        "perplexity": round(compute_perplexity(model, tokenizer, texts), 3),
    }
    if not args.skip_dialect_id:
        report["dialect_id"] = dialect_id_eval(model, tokenizer, rows, dialects)

    out = Path("reports")
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out / f"eval_{stamp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md = [f"# Lahja evaluation — {stamp}", "",
          f"- Model: `{report['model']}`",
          f"- Eval examples: {report['n_eval']}",
          f"- Perplexity: **{report['perplexity']}**"]
    if "dialect_id" in report:
        d = report["dialect_id"]
        md += [f"- Dialect ID accuracy: **{d['accuracy']}**",
               f"- Dialect ID macro-F1: **{d['macro_f1']}**"]
    (out / f"eval_{stamp}.md").write_text("\n".join(md))
    log.info("Report written → reports/eval_%s.{json,md}", stamp)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
