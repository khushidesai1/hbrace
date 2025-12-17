"""Sweep over different beta_t_l1_strength values."""
import os
import subprocess
import yaml
import shutil
import sys
import json
from datetime import datetime

# Beta_t L1 strength values to sweep
BETA_T_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]

# Base config - 1000 genes with 10% truly predictive
BASE_CONFIG = {
    "data": {
        "num_patients": 128,
        "test_fraction": 0.25,
        "batch_size": 8,
        "seed": 88,
        "device": "cpu",
        "beta_t_active_frac": 0.1,  # 10% of genes truly predictive
        "beta_t_active_scale": 2.0,
    },
    "model": {
        "n_subtypes": 3,
        "n_cell_types": 8,
        "n_genes": 1000,
        "z_dim": 8,
        "u_dim": 8,
        "subtype_concentration": 2.0,
        "subtype_rate": 0.1,
        "nb_dispersion_prior": 2.0,
        "nb_dispersion_rate": 1.0,
    },
    "vi": {
        "early_stopping_patience": 25,
        "learning_rate": 0.001,  # increased from 0.0005
        "num_epochs": 300,  # reduced from 500
        "log_interval": 100,
        "guide": "auto_normal",
    },
}


def run_single_experiment(beta_t: float):
    """Run training and evaluation for a single beta_t value."""
    run_name = f"sweep_beta_t_{beta_t}"
    print(f"\n{'='*60}")
    print(f"Running sweep with beta_t_l1_strength = {beta_t}")
    print(f"{'='*60}\n")

    # Create config for this run
    config = {
        "run_name": run_name,
        "data": BASE_CONFIG["data"].copy(),
        "model": BASE_CONFIG["model"].copy(),
        "vi": BASE_CONFIG["vi"].copy(),
    }
    config["model"]["beta_t_l1_strength"] = beta_t

    # Write to experiment.yaml (the default config path)
    config_path = "configs/experiment.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Clean up any existing data/checkpoints for fresh run
    data_path = f"./data/synthetic_data_{run_name}"
    checkpoint_path = f"saved_models/checkpoint_{run_name}.pth"

    # Remove old artifacts to ensure fresh run
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Run training
    print("Starting training...")
    train_result = subprocess.run(
        [sys.executable, "scripts/train_vi.py"],
        capture_output=False,
        text=True,
    )

    if train_result.returncode != 0:
        print(f"Training failed for beta_t = {beta_t}")
        return {"beta_t_l1_strength": beta_t, "success": False, "auprc": None}

    # Run evaluation
    print("\nStarting evaluation...")
    eval_result = subprocess.run(
        [sys.executable, "scripts/eval_model.py"],
        capture_output=True,
        text=True,
    )

    # Parse metrics from output
    auprc = None
    train_nll = None
    val_nll = None
    gen_gap = None

    for line in eval_result.stdout.split("\n"):
        if "AUPRC on validation" in line:
            try:
                auprc = float(line.split(":")[-1].strip())
            except:
                pass
        elif "Average validation NLL:" in line:
            try:
                val_nll = float(line.split(":")[1].split("+/-")[0].strip())
            except:
                pass
        elif "Average training NLL:" in line:
            try:
                train_nll = float(line.split(":")[1].split("+/-")[0].strip())
            except:
                pass
        elif "Generalization gap:" in line:
            try:
                gen_gap = float(line.split(":")[1].split("+/-")[0].strip())
            except:
                pass

    print(eval_result.stdout)
    if eval_result.returncode != 0:
        print(f"Evaluation errors: {eval_result.stderr}")

    # Build result dict
    result = {
        "beta_t_l1_strength": beta_t,
        "success": True,
        "auprc": auprc,
        "train_nll": train_nll,
        "val_nll": val_nll,
        "generalization_gap": gen_gap,
        "timestamp": datetime.now().isoformat(),
    }

    # Save individual run stats
    stats_path = f"results/{run_name}/stats.json"
    os.makedirs(f"results/{run_name}", exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved stats to {stats_path}")

    return result


def run_sweep():
    """Run training for each beta_t value."""
    results = []

    for beta_t in BETA_T_VALUES:
        result = run_single_experiment(beta_t)
        results.append(result)

    # Print summary
    print("\n" + "="*60)
    print("SWEEP SUMMARY")
    print("="*60)
    print(f"{'beta_t':>10} | {'Status':>8} | {'AUPRC':>8} | {'Val NLL':>12} | {'Gen Gap':>10}")
    print("-"*60)
    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        auprc = f"{r['auprc']:.3f}" if r.get("auprc") else "N/A"
        val_nll = f"{r['val_nll']:.1f}" if r.get("val_nll") else "N/A"
        gen_gap = f"{r['generalization_gap']:.1f}" if r.get("generalization_gap") else "N/A"
        print(f"{r['beta_t_l1_strength']:>10} | {status:>8} | {auprc:>8} | {val_nll:>12} | {gen_gap:>10}")

    # Save combined results
    with open("results/sweep_beta_t_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/sweep_beta_t_summary.json")

    return results


if __name__ == "__main__":
    # Allow running a single value from command line
    if len(sys.argv) > 1:
        beta_t = float(sys.argv[1])
        run_single_experiment(beta_t)
    else:
        run_sweep()
