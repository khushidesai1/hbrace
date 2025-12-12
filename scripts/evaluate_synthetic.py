"""Evaluation script for synthetic data experiments."""

import numpy as np
import torch
import pyro

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.patient_data import SimulatedDataGenerator
from hbrace.patient_data.dataset import get_train_test_dataloaders

# Patch pyro's param store to use weights_only=False for torch.load
import pyro.params.param_store as ps
original_load = ps.ParamStoreDict.load

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

print("Loading synthetic data...")
sim_data = SimulatedDataGenerator.load(data_path)

num_patients = sim_data.pre_counts.shape[0]
dataloader_train, dataloader_val = get_train_test_dataloaders(
    num_patients=num_patients,
    batch_size=data_config.batch_size,
    device=data_config.device,
    sim_data=sim_data,
    test_fraction=data_config.test_fraction,
)

print("Loading trained model...")
model = HBRACEModel(model_config, vi_config)

dummy_batch = next(iter(dataloader_train))
from hbrace.models.guides import build_guide
model.guide_fn = build_guide(model.model_fn, model.model_config, vi_config.guide)

model.load_checkpoint(checkpoint_path)

print("\nEvaluating predictive posterior...")

from hbrace.models.utils import predictive_log_likelihood

val_log_lik = predictive_log_likelihood(model.model_fn, model.guide_fn, dataloader_val, num_samples=32)
train_log_lik = predictive_log_likelihood(model.model_fn, model.guide_fn, dataloader_train, num_samples=32)

mean_val_nll = -val_log_lik
mean_train_nll = -train_log_lik
std_val_nll = 0.0
std_train_nll = 0.0

print(f"\nValidation NLL: {mean_val_nll:.2f} +/- {std_val_nll:.2f}")
print(f"Training NLL: {mean_train_nll:.2f} +/- {std_train_nll:.2f}")
print(f"Generalization gap: {mean_val_nll - mean_train_nll:.2f}")

guide_trace = pyro.poutine.trace(model.guide_fn).get_trace(dummy_batch)

print("\nLearned parameter shapes:")
for name in guide_trace.nodes.keys():
    if guide_trace.nodes[name]["type"] == "sample":
        value = guide_trace.nodes[name]["value"]
        print(f"  {name}: {tuple(value.shape)}")

print("\nGround truth parameters:")
if sim_data.extra_params:
    for key in sim_data.extra_params.keys():
        val = sim_data.extra_params[key]
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape}")
else:
    print("  No extra_params available")

param_store = pyro.get_param_store()
print(f"\nTotal parameters in store: {len(param_store)}")

param_names = list(param_store.keys())
for name in param_names[:5]:
    param_val = param_store[name]
    if isinstance(param_val, torch.Tensor):
        print(f"\n{name}:")
        print(f"  Shape: {tuple(param_val.shape)}")
        print(f"  Mean: {param_val.mean().item():.4f}, Std: {param_val.std().item():.4f}")
        print(f"  Range: [{param_val.min().item():.4f}, {param_val.max().item():.4f}]")
