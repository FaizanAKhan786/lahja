# Lahja | Quality & reproducibility

> **Status rule:** Check the [live GitHub Actions history](https://github.com/FaizanAKhan786/lahja/actions) for real workflow conclusions. This document does not invent benchmark scores or imply every feature was tested.

## Automated verification

Ruff lint and pytest on Python 3.10, 3.11 and 3.12, with LAHJA_MOCK_MODEL=1 (no model weights).

- [View the CI workflow](https://github.com/FaizanAKhan786/lahja/actions/workflows/ci.yml)
- Any failing run is an unresolved result until corrected; a green syntax/unit-check workflow is not proof of deployment readiness.

## Reproduce the checks

Run these commands from the **repository root**, unless the commands change directories:

```bash
pip install -e ".[dev]"
LAHJA_MOCK_MODEL=1 pytest -q
# For model evaluation, prepare data and train/merge a model first:
python -m lahja.eval.evaluate --config configs/training.yaml
```

## How to interpret the results

**Relevant metrics:** Held-out perplexity, dialect label accuracy, macro-F1 and optional judge-based linguistic checks.

**Requirements and limitations:** Provide a known base-model revision, split seed, evaluation file, merged model path and hardware. The mock API test does not assess linguistic quality.

Before comparing results, save the commit SHA, command, dataset and split (including seed), model revision, dependency environment, hardware/CPU/GPU, and the raw result file. Use the same evaluation set for all compared runs and distinguish **synthetic tests** from **real-world measurements**.

## Publication policy

Only publish numeric scores alongside the run configuration and raw evidence. A threshold, code assertion or example report must **not** be presented as a measured production result.

[Project README](../README.md) · [GitHub Actions](https://github.com/FaizanAKhan786/lahja/actions)
