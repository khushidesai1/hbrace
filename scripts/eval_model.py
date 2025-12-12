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
    logits = samples["logit_y"]  # (S, B)
    probs = torch.sigmoid(logits)

    y_obs = batch.responses  # (B,)
    var_term = (probs * (1 - probs)).clamp_min(eps)

    T_obs = ((y_obs - probs) ** 2 / var_term).sum(dim=1)  # (S,)
    y_rep = Bernoulli(probs).sample()
    T_rep = ((y_rep - probs) ** 2 / var_term).sum(dim=1)  # (S,)

    T_obs_total += T_obs
    T_rep_total += T_rep

ppc_p_value = (T_rep_total > T_obs_total).float().mean().item()
print(f"\nPosterior predictive p-value (chi-squared discrepancy on responses): {ppc_p_value:.3f}")


# %% Plot the simulations from the min X^2 distribution of y_rep and the min X^2 metric for observed value
