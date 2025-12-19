"""Gelman-style Posterior Predictive Checks for HBRACE model."""
# %% Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.models.guides import build_guide
from hbrace.models.ppc import (
    posterior_predictive_check_q_t,
)
from hbrace.patient_data import SimulatedDataGenerator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# %% Load configuration and data
config_path = "configs/experiment.yaml"
run_name, model_config, vi_config, data_config = load_config(config_path)

data_path = f"./data/synthetic_data_{run_name}"
checkpoint_path = f"saved_models/checkpoint_{run_name}.pth"

print("\nLoading synthetic data...")
sim_data = SimulatedDataGenerator.load(data_path)

# Check if extra_params are available
if sim_data.extra_params is None or len(sim_data.extra_params) == 0:
    print("\nWARNING: Regenerating data to get ground truth parameters...")
    torch.manual_seed(data_config.seed)
    generator = SimulatedDataGenerator.from_model_config(
        model_config=model_config,
        n_patients=data_config.num_patients,
        seed=data_config.seed,
        data_config=data_config,
    )
    _, sim_data = generator.generate_batch(
        device=data_config.device,
        return_simulation=True,
        save=True,
        name=data_path.split("/")[-1],
        out_dir="./data"
    )

num_patients = sim_data.pre_counts.shape[0]

# Create train/test split
print("Creating train/test split...")
train_indices, val_indices = train_test_split(
    np.arange(num_patients),
    test_size=data_config.test_fraction,
    random_state=data_config.seed,
)

# %% Load trained model
print("Loading trained model...")
model = HBRACEModel(model_config, vi_config)

# Initialize guide
train_batch = sim_data.to_patient_batch(device=data_config.device, indices=train_indices[:8])
model.guide_fn = build_guide(model.model_fn, model.model_config, vi_config.guide)

model.load_checkpoint(checkpoint_path)
print("Model loaded successfully.\n")

# %% Chi-squared discrepancy test
print("Computing chi-squared discrepancy on validation set...")
ppc_results = posterior_predictive_check_q_t(
    model=model,
    sim_data=sim_data,
    indices=val_indices,
    num_samples=500,
    device=data_config.device,
    batch_size=data_config.batch_size,
)

observed_stat = ppc_results["observed_statistic"]
replicated_stats = ppc_results["replicated_statistics"]
p_value = ppc_results["p_value"]

print(f"\nResults:")
print(f"Observed test statistic: {observed_stat:.2f}")
print(f"Mean replicated statistic: {replicated_stats.mean():.2f}")
print(f"Std replicated statistic: {replicated_stats.std():.2f}")
print(f"Bayesian p-value: {p_value:.3f}")

print(f"\nInterpretation:")
if 0.05 < p_value < 0.95:
    print(f"GOOD: p-value = {p_value:.3f} suggests good model fit.")
    print(f"The observed data is consistent with posterior predictive distribution.")
elif p_value <= 0.05:
    print(f"WARNING: p-value = {p_value:.3f} is very low.")
    print(f"Model may be underfitting - observed data is in the tail of predictive distribution.")
else:  # p_value >= 0.95
    print(f"WARNING: p-value = {p_value:.3f} is very high.")
    print(f"Model may be overfitting - observed data is too well explained.")

# %% Visualization 1: Histogram of test statistic (Gelman-style)
os.makedirs(f"results/{run_name}", exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Histogram of replicated statistics
ax.hist(replicated_stats, bins=50, alpha=0.7, color='steelblue',
        edgecolor='black', label='Replicated data')

# Vertical line for observed statistic
ax.axvline(observed_stat, color='red', linestyle='--', linewidth=3,
           label=f'Observed (p={p_value:.3f})')

# Add shaded regions for extreme values
percentile_5 = np.percentile(replicated_stats, 5)
percentile_95 = np.percentile(replicated_stats, 95)
ylim = ax.get_ylim()

# Shade extreme regions
ax.axvspan(replicated_stats.min(), percentile_5, alpha=0.2, color='red',
           label='Extreme (p<0.05 or p>0.95)')
ax.axvspan(percentile_95, replicated_stats.max(), alpha=0.2, color='red')

ax.set_xlabel('Test Statistic T(y)', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.set_title('Posterior Predictive Distribution of Test Statistic\n'
             f'(Bayesian p-value = {p_value:.3f})', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"results/{run_name}/ppc_gelman_histogram.png")
plt.savefig(f"results/{run_name}/ppc_gelman_histogram.svg")
print(f"\nSaved histogram to results/{run_name}/ppc_gelman_histogram.png")
