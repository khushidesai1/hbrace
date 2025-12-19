"""Causal inference evaluation for HBRACE model."""
# %% Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from pyro.infer import Predictive

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.models.guides import build_guide
from hbrace.models.causal_inference import CausalInferenceEvaluator
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

# Check if extra_params are available for ground truth validation
if sim_data.extra_params is None or len(sim_data.extra_params) == 0:
    print("\nWARNING: Regenerating data to get ground truth parameters for causal inference validation...")
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

# %% Get posterior samples on validation set
print("Sampling from posterior on validation set...")
n_posterior_samples = 1000

# Create validation dataloader
from torch.utils.data import DataLoader, TensorDataset

val_dataset = TensorDataset(
    torch.arange(len(val_indices), dtype=torch.long)
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=data_config.batch_size,
    shuffle=False
)

# Sample from posterior for each batch
return_sites = [
    "z", "u", "pi_p", "pi_t", "mu_t", "beta_t", "gamma",
    "beta_s", "beta0", "T", "W", "epsilon"
]
if model_config.composition_model == "poe":
    return_sites.append("V")

# Initialize containers for concatenated samples
posterior_samples = {site: [] for site in return_sites}

print(f"Running {n_posterior_samples} posterior samples across {len(val_dataloader)} batches...")

with torch.no_grad():
    for batch_idx_tensor, in val_dataloader:
        # Get actual patient indices for this batch
        batch_patient_indices = val_indices[batch_idx_tensor.numpy()]

        # Create batch
        batch = sim_data.to_patient_batch(
            device=data_config.device,
            indices=batch_patient_indices
        )

        # Sample from posterior for this batch
        predictive = Predictive(
            model.model_fn,
            guide=model.guide_fn,
            num_samples=n_posterior_samples,
            return_sites=return_sites,
            parallel=False,
        )

        batch_samples = predictive(batch)

        # Accumulate patient-specific samples
        for site in return_sites:
            if site in batch_samples:
                posterior_samples[site].append(batch_samples[site])

# Concatenate samples across batches
# Sites that are patient-specific need concatenation along patient dimension
patient_sites = ["z", "u", "pi_p", "pi_t", "mu_t", "epsilon"]
for site in return_sites:
    if site in posterior_samples and posterior_samples[site]:
        if site in patient_sites:
            # Concatenate along patient dimension (dim=1)
            posterior_samples[site] = torch.cat(posterior_samples[site], dim=1)
        else:
            # For global parameters, just take the first (they should be the same across batches)
            # and squeeze out any singleton dimensions
            posterior_samples[site] = posterior_samples[site][0]
            # Squeeze out singleton dimensions if present (e.g., beta_t might have extra dims)
            while posterior_samples[site].ndim > 2 and posterior_samples[site].shape[1] == 1:
                posterior_samples[site] = posterior_samples[site].squeeze(1)

print(f"Posterior sampling complete. Sample shapes:")
for key in ["z", "pi_t", "mu_t", "beta_t"]:
    if key in posterior_samples:
        print(f"  {key}: {posterior_samples[key].shape}")

# %% Initialize Causal Inference Evaluator
print("\nInitializing CausalInferenceEvaluator...")

# Create a SimulatedData object for validation set only
val_sim_data = type(sim_data)(
    config=sim_data.config,
    pre_counts=sim_data.pre_counts[val_indices],
    on_counts=sim_data.on_counts[val_indices],
    responses=sim_data.responses[val_indices],
    subtype_ids=sim_data.subtype_ids[val_indices],
    pi_p=sim_data.pi_p[val_indices],
    pi_t=sim_data.pi_t[val_indices],
    pre_cell_types=[sim_data.pre_cell_types[i] for i in val_indices] if sim_data.pre_cell_types else None,
    post_cell_types=[sim_data.post_cell_types[i] for i in val_indices] if sim_data.post_cell_types else None,
    extra_params={k: v[val_indices] if isinstance(v, np.ndarray) and len(v) == num_patients else v
                  for k, v in sim_data.extra_params.items()} if sim_data.extra_params else None
)

evaluator = CausalInferenceEvaluator(
    posterior_samples=posterior_samples,
    simulated_data=val_sim_data,
    config=model_config,
    device=data_config.device
)

print(f"Evaluator initialized with {evaluator.n_posterior_samples} posterior samples")
print(f"Number of validation patients: {evaluator.n_patients}")

# %% Example 1: Estimate ACE for on-treatment intervention with random delta per cell type
print("\n" + "="*80)
print("Estimating ACE for on-treatment interventions")
print("="*80)

# Sample random intervention magnitudes for each cell type
np.random.seed(42)  # For reproducibility
delta_range = (-3, 3)  # More moderate range than log(2)
delta_per_cell_type = np.random.uniform(delta_range[0], delta_range[1], size=model_config.n_cell_types)

print(f"\nIntervention magnitude range: δ ∈ [{delta_range[0]:.2f}, {delta_range[1]:.2f}]")
print(f"Sampled δ per cell type: {delta_per_cell_type}\n")

ace_results = {}
for cell_type_idx in range(model_config.n_cell_types):
    delta_t = delta_per_cell_type[cell_type_idx]
    result = evaluator.estimate_ACE_on_treatment(
        delta_t=delta_t,
        cell_type_idx=cell_type_idx,
        subtype=None
    )
    ace_results[cell_type_idx] = result

    print(f"Cell Type {cell_type_idx} (δ={delta_t:+.3f}):")
    print(f"  ACE: {result['ace_mean']:+.4f} [{result['ace_ci_lower']:+.4f}, {result['ace_ci_upper']:+.4f}]")
    print(f"  Std: {result['ace_std']:.4f}")

    # Print delta_sq diagnostic if available
    if hasattr(evaluator, '_delta_sq_diagnostic'):
        diag = evaluator._delta_sq_diagnostic
        print(f"  delta_sq diagnostic:")
        print(f"    Mean: {diag['mean']:.6f}, Std: {diag['std']:.6f}")
        print(f"    Range: [{diag['min']:.6f}, {diag['max']:.6f}]")
        print(f"    Median: {diag['median']:.6f}")

# %% Example 2: Subtype-specific effects
print("\n" + "="*80)
print("Estimating subtype-specific ACE")
print("="*80)

subtype_names = ['ER+', 'HER2+', 'TNBC'][:model_config.n_subtypes]
cell_type_idx = 0  # Focus on first cell type
delta_t_for_subtype = delta_per_cell_type[cell_type_idx]

print(f"\nIntervention: Cell type {cell_type_idx}, δ = {delta_t_for_subtype:+.3f}\n")

subtype_results = {}
for subtype in range(model_config.n_subtypes):
    result = evaluator.estimate_ACE_on_treatment(
        delta_t=delta_t_for_subtype,
        cell_type_idx=cell_type_idx,
        subtype=subtype
    )
    subtype_results[subtype] = result

    subtype_name = subtype_names[subtype] if subtype < len(subtype_names) else f"Subtype {subtype}"
    print(f"{subtype_name}:")
    print(f"  ACE: {result['ace_mean']:+.4f} [{result['ace_ci_lower']:+.4f}, {result['ace_ci_upper']:+.4f}]")

# %% Example 3: Dose-response curves
print("\n" + "="*80)
print("Computing dose-response curves")
print("="*80)

os.makedirs(f"results/{run_name}", exist_ok=True)

dose_response_results = {}
for cell_type_idx in range(model_config.n_cell_types):
    print(f"Computing dose-response for cell type {cell_type_idx}...")
    dose_response = evaluator.compute_dose_response_curve(
        cell_type_idx=cell_type_idx,
        delta_range=(-2.0, 2.0),
        n_points=20,
        subtype=None
    )
    dose_response_results[cell_type_idx] = dose_response

# Plot all dose-response curves
fig, axes = plt.subplots(1, model_config.n_cell_types, figsize=(5*model_config.n_cell_types, 5))
if model_config.n_cell_types == 1:
    axes = [axes]

for cell_type_idx, ax in enumerate(axes):
    dr = dose_response_results[cell_type_idx]

    ax.plot(dr['delta'], dr['ace_mean'], 'b-', linewidth=2, label='ACE')
    ax.fill_between(
        dr['delta'],
        dr['ace_ci_lower'],
        dr['ace_ci_upper'],
        alpha=0.3,
        label='95% CI'
    )
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    ax.set_xlabel('Intervention magnitude (δ)', fontsize=12)
    ax.set_ylabel('Average Causal Effect', fontsize=12)
    ax.set_title(f'Cell Type {cell_type_idx}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'results/{run_name}/causal_dose_response_curves.png', dpi=150, bbox_inches='tight')
plt.savefig(f'results/{run_name}/causal_dose_response_curves.svg')
print(f"\nSaved dose-response curves to results/{run_name}/causal_dose_response_curves.png")

# %% Example 4: Validate against ground truth (if available)
if evaluator.ground_truth:
    print("\n" + "="*80)
    print("Validating against ground truth")
    print("="*80)

    try:
        validation_results = {}
        delta_grid = np.linspace(-2.0, 2.0, 10)

        for cell_type_idx in range(model_config.n_cell_types):
            print(f"\nValidating cell type {cell_type_idx}...")
            validation = evaluator.validate_against_ground_truth(
                delta_grid=delta_grid,
                cell_type_idx=cell_type_idx,
                subtype=None
            )
            validation_results[cell_type_idx] = validation

            print(f"  MSE:         {validation['mse']:.6f}")
            print(f"  Coverage:    {validation['coverage']:.2%}")
            print(f"  Correlation: {validation['correlation']:.4f}")

        # Plot validation results
        fig, axes = plt.subplots(2, model_config.n_cell_types,
                                figsize=(5*model_config.n_cell_types, 10))
        if model_config.n_cell_types == 1:
            axes = axes.reshape(-1, 1)

        for cell_type_idx in range(model_config.n_cell_types):
            val = validation_results[cell_type_idx]

            # Top row: Estimated vs True ACE
            ax1 = axes[0, cell_type_idx]
            ax1.plot(val['delta_grid'], val['ace_estimated'], 'b-',
                    marker='o', label='Estimated', linewidth=2)
            ax1.plot(val['delta_grid'], val['ace_true'], 'r--',
                    marker='s', label='True', linewidth=2)
            ax1.fill_between(val['delta_grid'], val['ace_ci_lower'], val['ace_ci_upper'],
                            alpha=0.2, color='blue', label='95% CI')
            ax1.set_xlabel('Intervention (δ)', fontsize=12)
            ax1.set_ylabel('ACE', fontsize=12)
            ax1.set_title(f'Cell Type {cell_type_idx}\n(Correlation: {val["correlation"]:.3f})', fontsize=14)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Bottom row: Calibration plot
            ax2 = axes[1, cell_type_idx]
            ax2.scatter(val['ace_true'], val['ace_estimated'], alpha=0.6, s=80)
            lim_min = min(val['ace_true'].min(), val['ace_estimated'].min())
            lim_max = max(val['ace_true'].max(), val['ace_estimated'].max())
            ax2.plot([lim_min, lim_max], [lim_min, lim_max],
                    'k--', alpha=0.3, label='Perfect calibration')
            ax2.set_xlabel('True ACE', fontsize=12)
            ax2.set_ylabel('Estimated ACE', fontsize=12)
            ax2.set_title(f'Calibration (MSE: {val["mse"]:.4f})', fontsize=14)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/{run_name}/causal_validation_plot.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'results/{run_name}/causal_validation_plot.svg')
        print(f"\nSaved validation plots to results/{run_name}/causal_validation_plot.png")

    except Exception as e:
        print(f"\nGround truth validation failed: {e}")
        print("Skipping validation plots.")
else:
    print("\n" + "="*80)
    print("Ground truth not available - skipping validation")
    print("="*80)

# %% Example 5: Forest plot of ACE estimates
print("\n" + "="*80)
print("Creating forest plot of ACE estimates")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 6))

y_positions = []
labels = []
means = []
ci_lowers = []
ci_uppers = []

y_pos = 0
# Add overall effects for each cell type
for cell_type_idx in range(model_config.n_cell_types):
    result = ace_results[cell_type_idx]
    delta_val = delta_per_cell_type[cell_type_idx]
    y_positions.append(y_pos)
    labels.append(f'Cell Type {cell_type_idx} (δ={delta_val:+.3f})')
    means.append(result['ace_mean'])
    ci_lowers.append(result['ace_ci_lower'])
    ci_uppers.append(result['ace_ci_upper'])
    y_pos += 1

# Add spacing
y_pos += 0.5

# Add subtype-specific effects for first cell type
for subtype in range(model_config.n_subtypes):
    result = subtype_results[subtype]
    y_positions.append(y_pos)
    subtype_name = subtype_names[subtype] if subtype < len(subtype_names) else f"Subtype {subtype}"
    labels.append(f'  {subtype_name} (Cell 0)')
    means.append(result['ace_mean'])
    ci_lowers.append(result['ace_ci_lower'])
    ci_uppers.append(result['ace_ci_upper'])
    y_pos += 1

# Plot
for i, (y, mean, lower, upper) in enumerate(zip(y_positions, means, ci_lowers, ci_uppers)):
    color = 'steelblue' if i < model_config.n_cell_types else 'coral'
    ax.plot([lower, upper], [y, y], 'k-', linewidth=2)
    ax.plot(mean, y, 'o', color=color, markersize=10)

ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.set_yticks(y_positions)
ax.set_yticklabels(labels)
ax.set_xlabel('Average Causal Effect (ACE)', fontsize=12)
ax.set_title(f'Forest Plot of Causal Effects (δ sampled per cell type)', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'results/{run_name}/causal_forest_plot.png', dpi=150, bbox_inches='tight')
plt.savefig(f'results/{run_name}/causal_forest_plot.svg')
print(f"Saved forest plot to results/{run_name}/causal_forest_plot.png")

# %% Summary
print("SUMMARY")
print("\nCausal Effect Estimates:")
for cell_type_idx in range(model_config.n_cell_types):
    result = ace_results[cell_type_idx]
    delta_val = delta_per_cell_type[cell_type_idx]
    print(f"  Cell Type {cell_type_idx} (δ={delta_val:+.3f}): ACE = {result['ace_mean']:+.4f} "
          f"[{result['ace_ci_lower']:+.4f}, {result['ace_ci_upper']:+.4f}]")

print(f"\nSubtype Heterogeneity (Cell Type 0, δ={delta_t_for_subtype:+.3f}):")
for subtype in range(model_config.n_subtypes):
    result = subtype_results[subtype]
    subtype_name = subtype_names[subtype] if subtype < len(subtype_names) else f"Subtype {subtype}"
    print(f"  {subtype_name}: ACE = {result['ace_mean']:+.4f} "
          f"[{result['ace_ci_lower']:+.4f}, {result['ace_ci_upper']:+.4f}]")

if evaluator.ground_truth and validation_results:
    print("\nValidation Metrics:")
    for cell_type_idx in range(model_config.n_cell_types):
        val = validation_results[cell_type_idx]
        print(f"  Cell Type {cell_type_idx}:")
        print(f"    MSE: {val['mse']:.6f}")
        print(f"    Coverage: {val['coverage']:.2%}")
        print(f"    Correlation: {val['correlation']:.4f}")

print("Causal inference evaluation complete!")
