"""Posterior Predictive Check evaluation script for HBRACE model."""
# %% Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.models.guides import build_guide
from hbrace.models.ppc import (
    evaluate_q_t_reconstruction,
    evaluate_response_calibration,
    sample_q_t,
)
from hbrace.patient_data import SimulatedDataGenerator
from hbrace.patient_data.dataset import get_train_test_dataloaders

# %% Load configuration and data
config_path = "configs/experiment.yaml"
run_name, model_config, vi_config, data_config = load_config(config_path)

data_path = f"./data/synthetic_data_{run_name}"
checkpoint_path = f"saved_models/checkpoint_{run_name}.pth"

print("Loading synthetic data...")
sim_data = SimulatedDataGenerator.load(data_path)

# Check if extra_params are available
if sim_data.extra_params is None or len(sim_data.extra_params) == 0:
    print("\n" + "="*60)
    print("WARNING: extra_params not found in loaded data!")
    print("Regenerating data to get ground truth parameters...")
    print("="*60 + "\n")

    torch.manual_seed(data_config.seed)
    generator = SimulatedDataGenerator.from_model_config(
        model_config=model_config,
        n_patients=data_config.num_patients,
        seed=data_config.seed,
    )
    _, sim_data = generator.generate_batch(
        device=data_config.device,
        return_simulation=True,
        save=True,
        name=data_path.split("/")[-1],
        out_dir="./data"
    )
    print("Data regenerated with ground truth parameters.\n")

num_patients = sim_data.pre_counts.shape[0]

# Create train/test split
print("Creating train/test split...")
from sklearn.model_selection import train_test_split
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

# %% Evaluate q_t reconstruction on validation set
print("="*60)
print("POSTERIOR PREDICTIVE CHECK: q_t RECONSTRUCTION")
print("="*60)

print("\nEvaluating on VALIDATION set...")
val_global, val_per_patient, q_t_pred_val, q_t_true_val = evaluate_q_t_reconstruction(
    model=model,
    sim_data=sim_data,
    indices=val_indices,
    num_samples=1000,
    device=data_config.device,
    batch_size=data_config.batch_size,
)

print("\nGlobal Reconstruction Metrics (Validation):")
print(f"  MSE:         {val_global['mse']:.4f}")
print(f"  MAE:         {val_global['mae']:.4f}")
print(f"  R² score:    {val_global['r2']:.4f}")
print(f"  Pearson r:   {val_global['pearson_r']:.4f}")
print(f"  Spearman ρ:  {val_global['spearman_r']:.4f}")

print("\nPer-Patient Reconstruction Metrics (Validation):")
print(f"  MSE:         {val_per_patient['mse'].mean():.4f} ± {val_per_patient['mse'].std():.4f}")
print(f"  MAE:         {val_per_patient['mae'].mean():.4f} ± {val_per_patient['mae'].std():.4f}")
print(f"  R² score:    {val_per_patient['r2'].mean():.4f} ± {val_per_patient['r2'].std():.4f}")
print(f"  Pearson r:   {val_per_patient['pearson_r'].mean():.4f} ± {val_per_patient['pearson_r'].std():.4f}")
print(f"  Spearman ρ:  {val_per_patient['spearman_r'].mean():.4f} ± {val_per_patient['spearman_r'].std():.4f}")

# %% Evaluate on training set for comparison
print("\nEvaluating on TRAINING set...")
train_global, train_per_patient, q_t_pred_train, q_t_true_train = evaluate_q_t_reconstruction(
    model=model,
    sim_data=sim_data,
    indices=train_indices,
    num_samples=1000,
    device=data_config.device,
    batch_size=data_config.batch_size,
)

print("\nGlobal Reconstruction Metrics (Training):")
print(f"  MSE:         {train_global['mse']:.4f}")
print(f"  MAE:         {train_global['mae']:.4f}")
print(f"  R² score:    {train_global['r2']:.4f}")
print(f"  Pearson r:   {train_global['pearson_r']:.4f}")
print(f"  Spearman ρ:  {train_global['spearman_r']:.4f}")

# %% Visualization 1: Predicted vs True q_t (scatter plot)
os.makedirs(f"results/{run_name}", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Validation set
q_t_pred_val_mean = q_t_pred_val.mean(dim=0).cpu().numpy().flatten()
q_t_true_val_flat = q_t_true_val.flatten()

axes[0].scatter(q_t_true_val_flat, q_t_pred_val_mean, alpha=0.3, s=10)
axes[0].plot([q_t_true_val_flat.min(), q_t_true_val_flat.max()],
             [q_t_true_val_flat.min(), q_t_true_val_flat.max()],
             'r--', lw=2, label='Perfect reconstruction')
axes[0].set_xlabel('True q_t', fontsize=12)
axes[0].set_ylabel('Predicted q_t (mean)', fontsize=12)
axes[0].set_title(f'Validation Set (R²={val_global["r2"]:.3f})', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Training set
q_t_pred_train_mean = q_t_pred_train.mean(dim=0).cpu().numpy().flatten()
q_t_true_train_flat = q_t_true_train.flatten()

axes[1].scatter(q_t_true_train_flat, q_t_pred_train_mean, alpha=0.3, s=10)
axes[1].plot([q_t_true_train_flat.min(), q_t_true_train_flat.max()],
             [q_t_true_train_flat.max(), q_t_true_train_flat.max()],
             'r--', lw=2, label='Perfect reconstruction')
axes[1].set_xlabel('True q_t', fontsize=12)
axes[1].set_ylabel('Predicted q_t (mean)', fontsize=12)
axes[1].set_title(f'Training Set (R²={train_global["r2"]:.3f})', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"results/{run_name}/ppc_q_t_scatter.png", dpi=150)
plt.savefig(f"results/{run_name}/ppc_q_t_scatter.svg")
print(f"\nSaved scatter plot to results/{run_name}/ppc_q_t_scatter.png")

# %% Visualization 2: Residuals plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Validation residuals
residuals_val = q_t_pred_val_mean - q_t_true_val_flat
axes[0].scatter(q_t_true_val_flat, residuals_val, alpha=0.3, s=10)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('True q_t', fontsize=12)
axes[0].set_ylabel('Residual (Predicted - True)', fontsize=12)
axes[0].set_title(f'Validation Residuals (MAE={val_global["mae"]:.3f})', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Training residuals
residuals_train = q_t_pred_train_mean - q_t_true_train_flat
axes[1].scatter(q_t_true_train_flat, residuals_train, alpha=0.3, s=10)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('True q_t', fontsize=12)
axes[1].set_ylabel('Residual (Predicted - True)', fontsize=12)
axes[1].set_title(f'Training Residuals (MAE={train_global["mae"]:.3f})', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"results/{run_name}/ppc_q_t_residuals.png", dpi=150)
plt.savefig(f"results/{run_name}/ppc_q_t_residuals.svg")
print(f"Saved residual plot to results/{run_name}/ppc_q_t_residuals.png")

# %% Visualization 3: Per-patient correlation distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.hist(val_per_patient['pearson_r'], bins=20, alpha=0.5, label='Validation', color='blue')
ax.hist(train_per_patient['pearson_r'], bins=20, alpha=0.5, label='Training', color='orange')
ax.axvline(val_per_patient['pearson_r'].mean(), color='blue', linestyle='--', lw=2,
           label=f'Val mean={val_per_patient["pearson_r"].mean():.3f}')
ax.axvline(train_per_patient['pearson_r'].mean(), color='orange', linestyle='--', lw=2,
           label=f'Train mean={train_per_patient["pearson_r"].mean():.3f}')
ax.set_xlabel('Pearson correlation (per patient)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Per-Patient Correlations', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"results/{run_name}/ppc_per_patient_corr.png", dpi=150)
plt.savefig(f"results/{run_name}/ppc_per_patient_corr.svg")
print(f"Saved per-patient correlation plot to results/{run_name}/ppc_per_patient_corr.png")

# %% Visualization 4: Uncertainty quantification
# Show posterior uncertainty for a few example patients
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

n_examples = min(6, len(val_indices))
example_patients = np.random.choice(len(val_indices), n_examples, replace=False)

for idx, patient_idx in enumerate(example_patients):
    ax = axes[idx]

    # Get predictions for this patient across all genes
    pred_samples = q_t_pred_val[:, patient_idx, :].cpu().numpy()  # (num_samples, G)
    true_vals = q_t_true_val[patient_idx, :]  # (G,)

    # Compute mean and credible intervals
    pred_mean = pred_samples.mean(axis=0)
    pred_lower = np.percentile(pred_samples, 2.5, axis=0)
    pred_upper = np.percentile(pred_samples, 97.5, axis=0)

    # Sort by true value for better visualization
    sort_idx = np.argsort(true_vals)
    gene_idx = np.arange(len(true_vals))

    ax.plot(gene_idx, true_vals[sort_idx], 'r-', lw=2, label='True', alpha=0.7)
    ax.plot(gene_idx, pred_mean[sort_idx], 'b-', lw=2, label='Predicted (mean)', alpha=0.7)
    ax.fill_between(gene_idx, pred_lower[sort_idx], pred_upper[sort_idx],
                     alpha=0.3, label='95% CI')

    corr = val_per_patient['pearson_r'][patient_idx]
    ax.set_title(f'Patient {patient_idx} (r={corr:.3f})', fontsize=12)
    ax.set_xlabel('Gene (sorted by true value)', fontsize=10)
    ax.set_ylabel('q_t', fontsize=10)
    if idx == 0:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"results/{run_name}/ppc_uncertainty.png", dpi=150)
plt.savefig(f"results/{run_name}/ppc_uncertainty.svg")
print(f"Saved uncertainty plot to results/{run_name}/ppc_uncertainty.png")

# %% Summary
print("\n" + "="*60)
print("POSTERIOR PREDICTIVE CHECK SUMMARY")
print("="*60)
print(f"\nq_t Reconstruction Performance:")
print(f"  Validation R²:   {val_global['r2']:.4f}")
print(f"  Training R²:     {train_global['r2']:.4f}")
print(f"  Generalization:  {train_global['r2'] - val_global['r2']:.4f} (train - val)")
print(f"\n  Validation Pearson r:  {val_global['pearson_r']:.4f}")
print(f"  Training Pearson r:    {train_global['pearson_r']:.4f}")

print(f"\nInterpretation:")
if val_global['r2'] > 0.8:
    print("  ✓ Excellent reconstruction! Model accurately captures q_t.")
elif val_global['r2'] > 0.6:
    print("  ✓ Good reconstruction. Model captures most variation in q_t.")
elif val_global['r2'] > 0.4:
    print("  ~ Moderate reconstruction. Some improvement needed.")
else:
    print("  ✗ Poor reconstruction. Model struggles to capture q_t variation.")
    print("    Consider: longer training, different guide, or checking for")
    print("    remaining prior mismatches.")

print("\n" + "="*60)
