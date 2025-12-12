"""Posterior predictive checks for model validation."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
from pyro import poutine

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.patient_data import SimulatedDataGenerator
from hbrace.patient_data.dataset import get_train_test_dataloaders

import pyro.params.param_store as ps

def patched_load(self, filename, map_location=None):
    with open(filename, 'rb') as input_file:
        state = torch.load(input_file, map_location, weights_only=False)
    params = state.get("params", {})
    constraints = state.get("constraints", {})
    for name, value in params.items():
        self._params[name] = value.contiguous()
    for name, constraint in constraints.items():
        self._constraints[name] = constraint

ps.ParamStoreDict.load = patched_load

data_path = "./data/synthetic_data_test"
config_path = "configs/experiment.yaml"
checkpoint_path = "saved_models/checkpoint.pth"

model_config, vi_config, data_config = load_config(config_path)

print("Loading data and model...")
sim_data = SimulatedDataGenerator.load(data_path)

num_patients = sim_data.pre_counts.shape[0]
dataloader_train, dataloader_val = get_train_test_dataloaders(
    num_patients=num_patients,
    batch_size=data_config.batch_size,
    device=data_config.device,
    sim_data=sim_data,
    test_fraction=data_config.test_fraction,
)

model = HBRACEModel(model_config, vi_config)
dummy_batch = next(iter(dataloader_train))
from hbrace.models.guides import build_guide
model.guide_fn = build_guide(model.model_fn, model.model_config, vi_config.guide)
model.load_checkpoint(checkpoint_path)

print("Computing held-out predictive likelihoods...")
from hbrace.models.utils import predictive_log_likelihood

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_log_liks = []
val_log_liks = []
sample_sizes = [4, 8, 16, 32, 64]

for n in sample_sizes:
    train_ll = predictive_log_likelihood(model.model_fn, model.guide_fn,
                                         dataloader_train, num_samples=n)
    val_ll = predictive_log_likelihood(model.model_fn, model.guide_fn,
                                       dataloader_val, num_samples=n)
    train_log_liks.append(-train_ll)
    val_log_liks.append(-val_ll)
    print(f"n={n}: train NLL={-train_ll:.2f}, val NLL={-val_ll:.2f}")

axes[0].plot(sample_sizes, train_log_liks, 'o-', linewidth=2, markersize=8, label='Training')
axes[0].plot(sample_sizes, val_log_liks, 's-', linewidth=2, markersize=8, label='Validation')
axes[0].set_xlabel('Number of posterior samples')
axes[0].set_ylabel('Negative log likelihood')
axes[0].set_title('Held-out predictive likelihood')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

gap = np.array(val_log_liks) - np.array(train_log_liks)
axes[1].plot(sample_sizes, gap, 'o-', linewidth=2, markersize=8, color='green')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Number of posterior samples')
axes[1].set_ylabel('Generalization gap (val NLL - train NLL)')
axes[1].set_title('Model generalization assessment')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/held_out_predictive_checks.png', dpi=150)
print("\nSaved held_out_predictive_checks.png")

print("\nGenerating posterior parameter summaries...")
guide_trace = poutine.trace(model.guide_fn).get_trace(dummy_batch)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

param_samples = {}
for name in ['phi_p_std', 'beta0', 'lambda_T', 'beta_t']:
    if name in guide_trace.nodes and guide_trace.nodes[name]["type"] == "sample":
        val = guide_trace.nodes[name]["value"].detach().cpu().numpy()
        param_samples[name] = val

if 'phi_p_std' in param_samples:
    axes2[0, 0].bar(range(len(param_samples['phi_p_std'])), param_samples['phi_p_std'])
    axes2[0, 0].set_xlabel('Cell type')
    axes2[0, 0].set_ylabel('Dispersion std')
    axes2[0, 0].set_title('Learned pre-treatment NB dispersion (phi_p_std)')
    axes2[0, 0].grid(True, alpha=0.3, axis='y')

if 'beta0' in param_samples:
    axes2[0, 1].text(0.5, 0.5, f'beta0 = {param_samples["beta0"]:.3f}',
                     ha='center', va='center', fontsize=16,
                     transform=axes2[0, 1].transAxes)
    axes2[0, 1].set_title('Response model intercept')
    axes2[0, 1].axis('off')

if 'lambda_T' in param_samples:
    axes2[1, 0].text(0.5, 0.5, f'lambda_T = {param_samples["lambda_T"]:.3f}',
                     ha='center', va='center', fontsize=16,
                     transform=axes2[1, 0].transAxes)
    axes2[1, 0].set_title('Cell composition shift scale')
    axes2[1, 0].axis('off')

if 'beta_t' in param_samples:
    axes2[1, 1].bar(range(len(param_samples['beta_t'])), param_samples['beta_t'])
    axes2[1, 1].set_xlabel('Cell type')
    axes2[1, 1].set_ylabel('Response coefficient')
    axes2[1, 1].set_title('Cell type effects on response (beta_t)')
    axes2[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/learned_parameters.png', dpi=150)
print("Saved learned_parameters.png")

print("\nAll plots saved to results/")
