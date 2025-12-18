"""
Diagnostic script to debug causal inference issues.
Checks parameter recovery, scaling, and model calibration.
"""
# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.models.guides import build_guide
from hbrace.patient_data import SimulatedDataGenerator

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# %% Load configuration and data
config_path = "configs/experiment.yaml"
run_name, model_config, vi_config, data_config = load_config(config_path)

data_path = f"./data/synthetic_data_{run_name}"
checkpoint_path = f"saved_models/checkpoint_{run_name}.pth"

print("\nLoading synthetic data...")
sim_data = SimulatedDataGenerator.load(data_path)

# Ensure we have ground truth
if sim_data.extra_params is None or len(sim_data.extra_params) == 0:
    print("\nRegenerating data to get ground truth...")
    torch.manual_seed(data_config.seed)
    generator = SimulatedDataGenerator.from_model_config(
        model_config=model_config,
        n_patients=data_config.num_patients,
        seed=data_config.seed,
        data_config=data_config,  # CRITICAL: Pass data_config for proper parameters!
    )
    _, sim_data = generator.generate_batch(
        device=data_config.device,
        return_simulation=True,
        save=True,
        name=data_path.split("/")[-1],
        out_dir="./data"
    )

print(f"Loaded {sim_data.pre_counts.shape[0]} patients")

# %% 1. Check ground truth parameters
print("\n" + "="*80)
print("1. GROUND TRUTH PARAMETER INSPECTION")
print("="*80)

if sim_data.extra_params:
    # Check key causal parameters
    if 'beta_t' in sim_data.extra_params:
        beta_t_true = sim_data.extra_params['beta_t']
        print(f"\nbeta_t (gene->response effects):")
        print(f"  Range: [{beta_t_true.min():.4f}, {beta_t_true.max():.4f}]")
        print(f"  Mean: {beta_t_true.mean():.4f}, Std: {beta_t_true.std():.4f}")
        print(f"  Non-zero: {(np.abs(beta_t_true) > 0.01).sum()} / {len(beta_t_true)}")

    if 'gamma' in sim_data.extra_params:
        gamma_true = sim_data.extra_params['gamma']
        print(f"\ngamma (confounder->response effects):")
        print(f"  Range: [{gamma_true.min():.4f}, {gamma_true.max():.4f}]")
        print(f"  Mean: {gamma_true.mean():.4f}, Std: {gamma_true.std():.4f}")

    if 'T' in sim_data.extra_params:
        T_true = sim_data.extra_params['T']
        print(f"\nT (composition interaction matrix):")
        print(f"  Range: [{T_true.min():.4f}, {T_true.max():.4f}]")
        print(f"  Mean: {T_true.mean():.4f}, Std: {T_true.std():.4f}")
        print(f"  Sparsity: {(np.abs(T_true) < 0.01).sum()} / {T_true.size} are near-zero")

else:
    print("WARNING: No ground truth parameters available!")

# %% 2. Check response distribution
print("\n" + "="*80)
print("2. RESPONSE DISTRIBUTION")
print("="*80)

responses = sim_data.responses
print(f"\nResponse statistics:")
print(f"  Mean: {responses.mean():.4f}")
print(f"  Class balance: {responses.sum()}/{len(responses)} = {responses.mean():.1%} positive")

# Check if model can even predict responses
print(f"\nBaseline accuracy (always predict majority): {max(responses.mean(), 1-responses.mean()):.1%}")

# %% 3. Check composition ranges
print("\n" + "="*80)
print("3. COMPOSITION STATISTICS")
print("="*80)

print(f"\nPre-treatment compositions (pi_p):")
print(f"  Shape: {sim_data.pi_p.shape}")
print(f"  Range: [{sim_data.pi_p.min():.4f}, {sim_data.pi_p.max():.4f}]")
print(f"  Per-cell-type means: {sim_data.pi_p.mean(axis=0)}")

print(f"\nOn-treatment compositions (pi_t):")
print(f"  Shape: {sim_data.pi_t.shape}")
print(f"  Range: [{sim_data.pi_t.min():.4f}, {sim_data.pi_t.max():.4f}]")
print(f"  Per-cell-type means: {sim_data.pi_t.mean(axis=0)}")

# Check how much compositions change with treatment
composition_change = np.abs(sim_data.pi_t - sim_data.pi_p).mean(axis=0)
print(f"\nAverage composition shift per cell type:")
for c, shift in enumerate(composition_change):
    print(f"  Cell type {c}: {shift:.4f}")

# %% 4. Load model and check posterior
print("\n" + "="*80)
print("4. POSTERIOR PARAMETER RECOVERY")
print("="*80)

print("Loading trained model...")
model = HBRACEModel(model_config, vi_config)

# Get a small batch to initialize guide
train_indices, val_indices = train_test_split(
    np.arange(sim_data.pre_counts.shape[0]),
    test_size=data_config.test_fraction,
    random_state=data_config.seed,
)
train_batch = sim_data.to_patient_batch(device=data_config.device, indices=train_indices[:8])
model.guide_fn = build_guide(model.model_fn, model.model_config, vi_config.guide)

model.load_checkpoint(checkpoint_path)
print("Model loaded.\n")

# Sample from posterior on a subset
from pyro.infer import Predictive

n_samples = 100
# Use the same batch size as training to avoid shape mismatches
batch_size = data_config.batch_size
subset_indices = train_indices[:batch_size]
subset_batch = sim_data.to_patient_batch(device=data_config.device, indices=subset_indices)

predictive = Predictive(
    model.model_fn,
    guide=model.guide_fn,
    num_samples=n_samples,
    return_sites=["beta_t", "gamma", "T", "W", "beta0"],
    parallel=False,
)

print(f"Sampling {n_samples} posterior samples on batch of {batch_size} patients...")
with torch.no_grad():
    posterior = predictive(subset_batch)

# Compare posterior to ground truth
if 'beta_t' in posterior and 'beta_t' in sim_data.extra_params:
    beta_t_post = posterior['beta_t'].cpu().numpy()
    # Squeeze out singleton dims
    while beta_t_post.ndim > 2 and beta_t_post.shape[1] == 1:
        beta_t_post = np.squeeze(beta_t_post, axis=1)

    beta_t_true = sim_data.extra_params['beta_t']
    beta_t_est = beta_t_post.mean(axis=0)

    print(f"beta_t comparison:")
    print(f"  True - Range: [{beta_t_true.min():.4f}, {beta_t_true.max():.4f}], Mean: {beta_t_true.mean():.4f}")
    print(f"  Posterior - Range: [{beta_t_est.min():.4f}, {beta_t_est.max():.4f}], Mean: {beta_t_est.mean():.4f}")
    print(f"  Correlation: {np.corrcoef(beta_t_true, beta_t_est)[0, 1]:.4f}")

if 'gamma' in posterior and 'gamma' in sim_data.extra_params:
    gamma_post = posterior['gamma'].cpu().numpy()
    gamma_true = sim_data.extra_params['gamma']
    gamma_est = gamma_post.mean(axis=0)

    print(f"\ngamma comparison:")
    print(f"  True: {gamma_true}")
    print(f"  Posterior mean: {gamma_est}")
    print(f"  Correlation: {np.corrcoef(gamma_true, gamma_est)[0, 1]:.4f}")

# %% 5. Check prediction calibration
print("\n" + "="*80)
print("5. RESPONSE PREDICTION CALIBRATION")
print("="*80)

# Get predictions on validation set - need to batch this too
from torch.utils.data import DataLoader, TensorDataset

val_dataset = TensorDataset(torch.arange(len(val_indices), dtype=torch.long))
val_dataloader = DataLoader(val_dataset, batch_size=data_config.batch_size, shuffle=False)

all_logits = []
n_pred_samples = 100

print(f"Sampling predictions on {len(val_indices)} validation patients...")
with torch.no_grad():
    for batch_idx_tensor, in val_dataloader:
        batch_patient_indices = val_indices[batch_idx_tensor.numpy()]
        batch = sim_data.to_patient_batch(device=data_config.device, indices=batch_patient_indices)

        predictive_y = Predictive(
            model.model_fn,
            guide=model.guide_fn,
            num_samples=n_pred_samples,
            return_sites=["logit_y"],
            parallel=False,
        )

        preds = predictive_y(batch)
        all_logits.append(preds['logit_y'].cpu())

# Concatenate across batches
logits = torch.cat(all_logits, dim=1).numpy()  # (n_samples, n_patients)

print(f"\nLogit statistics:")
print(f"  Shape: {logits.shape}")
print(f"  Range: [{logits.min():.2f}, {logits.max():.2f}]")
print(f"  Mean: {logits.mean():.2f}, Std: {logits.std():.2f}")

# Check for overflow issues
if np.abs(logits).max() > 20:
    print(f"  WARNING: Very large logits detected! This will cause overflow in sigmoid.")
    print(f"  Consider adjusting logit_scale or head_input_scale")

probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))  # Clip to avoid overflow
prob_mean = probs.mean(axis=0)

y_true_val = sim_data.responses[val_indices]

# Calibration plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Predicted vs actual
axes[0].scatter(prob_mean, y_true_val, alpha=0.6)
axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
axes[0].set_xlabel('Predicted Probability')
axes[0].set_ylabel('True Response')
axes[0].set_title('Prediction Calibration')
axes[0].grid(True, alpha=0.3)

# Distribution of predictions by class
axes[1].hist(prob_mean[y_true_val == 0], bins=20, alpha=0.5, label='Non-responders', density=True)
axes[1].hist(prob_mean[y_true_val == 1], bins=20, alpha=0.5, label='Responders', density=True)
axes[1].set_xlabel('Predicted Probability')
axes[1].set_ylabel('Density')
axes[1].set_title('Prediction Distribution by Class')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'results/{run_name}/diagnostic_calibration.png', dpi=150)
print(f"\nSaved calibration plot to results/{run_name}/diagnostic_calibration.png")

# %% 6. Check config settings
print("\n" + "="*80)
print("6. MODEL CONFIGURATION")
print("="*80)

print(f"\nKey configuration settings:")
print(f"  logit_scale: {model_config.logit_scale}")
print(f"  head_input_scale: {model_config.head_input_scale}")
print(f"  beta0_loc: {model_config.beta0_loc}")
print(f"  beta0_scale: {model_config.beta0_scale}")
print(f"  gene_sparsity: {model_config.gene_sparsity}")
print(f"  composition_model: {model_config.composition_model}")

# Check if these settings might be causing issues
if model_config.head_input_scale < 0.01:
    print(f"  WARNING: head_input_scale is very small! This will make inputs to the response model tiny.")
if model_config.logit_scale < 0.1:
    print(f"  WARNING: logit_scale is very small! This will make predicted effects tiny.")

# %% 7. Recommendations
print("\n" + "="*80)
print("7. DEBUGGING RECOMMENDATIONS")
print("="*80)

issues = []

# Check coverage
if 'validation_results' in locals():
    avg_coverage = np.mean([v['coverage'] for v in validation_results.values()])
    if avg_coverage < 0.5:
        issues.append("Poor coverage (<50%) suggests credible intervals are too narrow or model misspecification")

# Check correlations
if 'validation_results' in locals():
    neg_corrs = [k for k, v in validation_results.items() if v['correlation'] < 0]
    if neg_corrs:
        issues.append(f"Negative correlations for cell types {neg_corrs} suggest model is learning wrong direction")

# Check ACE magnitudes
if np.abs(logits).max() > 20:
    issues.append("Logit overflow detected - reduce logit_scale or head_input_scale")

if issues:
    print("\nPotential issues detected:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
else:
    print("\nNo obvious issues detected in this diagnostic.")

print("\nSuggested debugging steps:")
print("1. Check if beta_t is being recovered (look at correlation above)")
print("2. Verify logit_scale and head_input_scale aren't too small/large")
print("3. Check if ground truth ACE values are also small (might be data issue)")
print("4. Try training with different seeds to check stability")
print("5. Visualize dose-response curves to see if monotonic trends exist")

print("\n" + "="*80)
print("Diagnostic complete!")
print("="*80)
