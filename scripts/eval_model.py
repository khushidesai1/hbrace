# %% Import necessary libraries
import os
import pyro
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Bernoulli

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.patient_data import SimulatedDataGenerator
from hbrace.patient_data.dataset import get_train_test_dataloaders

from hbrace.models.utils import auprc_for_responses, predictive_log_likelihood, posterior_predictive_check
from hbrace.models.guides import build_guide
from sklearn.metrics import precision_recall_curve

# %% Load the data and the model
config_path = "configs/experiment.yaml"
run_name, model_config, vi_config, data_config = load_config(config_path)

data_path = f"./data/synthetic_data_{run_name}"
checkpoint_path = f"saved_models/checkpoint_{run_name}.pth"

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
    oversample=False,
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
sample_sizes = [64, 128, 256]

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

# %% AUPRC for response prediction on validation set
n_samples = 1000
auprc, y_true, y_score = auprc_for_responses(
    model.model_fn,
    model.guide_fn,
    dataloader_val,
    num_samples=n_samples,
    device=torch.device(data_config.device),
)
print(f"AUPRC on validation responses: {auprc:.3f}")

# %% Plot the PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
plt.figure()
plt.plot(recall, precision, label="AUPRC")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.legend()
plt.savefig(f"results/{run_name}/pr_curve.png")

# %% Posterior predictive check (chi-squared discrepancy on counts)
print("\nPosterior predictive check (on pre/post count distributions)...")
ppc_samples = 200
p_value, T_obs, T_rep = posterior_predictive_check(
    model.model_fn,
    model.guide_fn,
    dataloader_val,
    num_samples=ppc_samples,
    device=torch.device(data_config.device),
    target="counts",  # chi-squared on pre/post count distributions
)
print(f"Posterior predictive p-value (chi-squared on counts): {p_value:.3f}")
print(f"  (Good model fit: p-value close to 0.5, extreme values indicate misfit)")

# %% Plot histogram of T_rep vs T_obs (Fig 3 style from Gelman et al. 1996)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(T_obs, bins=30, alpha=0.7, label="T(y_obs)", color="blue")
plt.hist(T_rep, bins=30, alpha=0.7, label="T(y_rep)", color="orange")
plt.xlabel("Chi-squared discrepancy")
plt.ylabel("Frequency")
plt.title("Posterior Predictive Check")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(T_obs, T_rep, alpha=0.3, s=10)
max_val = max(T_obs.max(), T_rep.max())
plt.plot([0, max_val], [0, max_val], "r--", label="y=x")
plt.xlabel("T(y_obs, θ)")
plt.ylabel("T(y_rep, θ)")
plt.title(f"p-value = {p_value:.3f}")
plt.legend()

plt.tight_layout()
plt.savefig(f"results/{run_name}/ppc_histogram.png")
print(f"Saved PPC histogram to results/{run_name}/ppc_histogram.png")