# Hierarchical Breast Cancer Response Analysis (HBRACE)

An application of the hierarchical causal model inspired by the CAIRE formulation in [Wienstein et. al, 2024] for estimating patient response in breast cancer T cells upon receiving treatment of anti-PD1. This method builds on the idea that T cell compositions can heavily shape patient response. We collapse a hierarchical model to a patient level causal graph and use causal inference to analyze the effects of perturbing either the pre-treatment or post-treatment T cell composition of a patient for different subtypes of breast cancer.

## Installation (conda + editable pip)
- Create and activate an environment with Python 3.10+: `conda create -n hbrace python=3.10 && conda activate hbrace`.
- Install the project in editable mode from the repo root so scripts import `hbrace`: `pip install -e .` (or `pip install -e .[dev]` for lint/format tools).

## Training Configuration
- All training/evaluation scripts read `configs/experiment.yaml`; update this file (e.g., `run_name`, `model`, `vi`, `data` blocks) before running to set seeds, device, and hyperparameters.
- Kick off training with the current config via `python scripts/train_vi.py`, which will generate synthetic data under `data/`, save run configs in `results/<run_name>/`, and write checkpoints to `saved_models/`.

## Evaluation
- Key entry points: `scripts/eval_model.py` (predictive metrics/PR curve), `scripts/eval_causal_inference.py` (ACE estimates/plots), `scripts/eval_ppc_gelman.py`, and diagnostics in `scripts/diagnose_*.py`.
- After training, run them directly from the repo root (e.g., `python scripts/eval_model.py`); they reuse the active `configs/experiment.yaml` and corresponding checkpoint.
