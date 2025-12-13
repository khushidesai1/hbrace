# %% Import necessary libraries
import pyro
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Bernoulli

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.patient_data import SimulatedDataGenerator
from hbrace.patient_data.dataset import get_train_test_dataloaders
from hbrace.models.utils import predictive_log_likelihood
from hbrace.models.guides import build_guide

# %% Load the data and the model
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
    seed=data_config.seed,
)

print("Loading trained model...")
model = HBRACEModel(model_config, vi_config)

dummy_batch = next(iter(dataloader_train))
model.guide_fn = build_guide(model.model_fn, model.model_config, vi_config.guide)

model.load_checkpoint(checkpoint_path)

# %% Compute the discrepancy metric
print("\nEvaluating predictive posterior...")

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

train_log_liks = np.array(train_log_liks)
val_log_liks = np.array(val_log_liks)
print(f"Average validation NLL: {np.mean(val_log_liks):.2f} +/- {np.std(val_log_liks):.2f}")
print(f"Average training NLL: {np.mean(train_log_liks):.2f} +/- {np.std(train_log_liks):.2f}")
print(f"Generalization gap: {np.mean(val_log_liks) - np.mean(train_log_liks):.2f} +/- {np.std(val_log_liks - train_log_liks):.2f}")

# %% Compute T statistics based on the discrepancy from Gelman et al. 1996
ppc_samples = 128
eps = 1e-6

T_obs_total = torch.zeros(ppc_samples, device=data_config.device)
T_rep_total = torch.zeros(ppc_samples, device=data_config.device)

for batch in dataloader_val:
    batch = batch.to(data_config.device)
    predictive = pyro.infer.Predictive(
        model.model_fn,
        guide=model.guide_fn,
        num_samples=ppc_samples,
        return_sites=("logit_y",),
        parallel=False,
    )
    samples = predictive(batch)
    logits = samples["logit_y"]  # (S, B) or possibly (S, B, ...)
    probs = torch.sigmoid(logits)

    # Align observed responses with predictive samples, allowing for extra event dims.
    y_obs = batch.responses
    while y_obs.dim() < probs.dim() - 1:
        y_obs = y_obs.unsqueeze(-1)
    y_obs = y_obs.unsqueeze(0).expand_as(probs)

    var_term = (probs * (1 - probs)).clamp_min(eps)

    # Collapse all non-sample dims for the chi-squared discrepancy.
    T_obs = ((y_obs - probs) ** 2 / var_term).flatten(start_dim=1).sum(dim=1)  # (S,)
    y_rep = Bernoulli(probs).sample()
    T_rep = ((y_rep - probs) ** 2 / var_term).flatten(start_dim=1).sum(dim=1)  # (S,)

    T_obs_total += T_obs
    T_rep_total += T_rep

ppc_p_value = (T_rep_total > T_obs_total).float().mean().item()
print(f"\nPosterior predictive p-value (chi-squared discrepancy on responses): {ppc_p_value:.3f}")

# Baseline null (global-mean Bernoulli) PPC on responses
all_val_batches = [b.to(data_config.device) for b in dataloader_val]
val_responses = torch.cat([b.responses for b in all_val_batches], dim=0)
global_mean = val_responses.mean().clamp(0.0, 1.0)

null_probs = torch.full((ppc_samples, val_responses.shape[0]), global_mean, device=data_config.device)
null_var = (null_probs * (1 - null_probs)).clamp_min(eps)
null_y_obs = val_responses.unsqueeze(0).expand_as(null_probs)
null_T_obs = ((null_y_obs - null_probs) ** 2 / null_var).sum(dim=1)
null_y_rep = Bernoulli(null_probs).sample()
null_T_rep = ((null_y_rep - null_probs) ** 2 / null_var).sum(dim=1)
null_p_value = (null_T_rep > null_T_obs).float().mean().item()
print(f"Baseline null p-value (global-mean Bernoulli): {null_p_value:.3f}")

# %% Plot the simulations from the min X^2 distribution of y_rep and the min X^2 metric for observed value
