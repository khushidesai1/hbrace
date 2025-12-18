# %% Import necessary libraries
import numpy as np
import torch
import pyro
from pyro.infer import Predictive

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.patient_data import SimulatedDataGenerator

# %% Load configuration
config_path = "results/best_model_even_bigger_shift/config.yaml"
run_name, model_config, vi_config, data_config = load_config(config_path)

# %% Load data
data_path = f"./data/synthetic_data_{run_name}"
print(f"Loading data from {data_path}")
sim_data = SimulatedDataGenerator.load(data_path)
batch = sim_data.to_patient_batch(device=data_config.device)

# %% Load trained model
print("Loading trained model...")
model = HBRACEModel(model_config, vi_config)
checkpoint_path = f"saved_models/checkpoint_{run_name}.pth"
model.load_checkpoint(checkpoint_path, map_location=data_config.device)

# %% Build guide
from hbrace.models.guides import build_guide
model.guide_fn = build_guide(
    model.model_fn,
    model_config,
    vi_config.guide,
    rank=vi_config.guide_rank,
)

# %% Sample from posterior with more samples to estimate uncertainty
print("\nSampling from posterior to estimate uncertainty...")
num_samples = 500
pyro.set_rng_seed(42)

# %% Process first batch only for speed
batch_size = 8
mini_batch = batch
mini_batch.pre_counts = batch.pre_counts[:batch_size]
mini_batch.on_counts = batch.on_counts[:batch_size]
mini_batch.cell_type_proportions = batch.cell_type_proportions[:batch_size]
mini_batch.responses = batch.responses[:batch_size]
mini_batch.subtype_ids = batch.subtype_ids[:batch_size]

with torch.no_grad():
    predictive = Predictive(model.model_fn, guide=model.guide_fn, num_samples=num_samples)
    samples = predictive(mini_batch)

# %% Analyze z uncertainty
z_samples = samples["z"].cpu().numpy()  # (num_samples, batch_size, d_z)
print(f"\nz samples shape: {z_samples.shape}")

z_mean = z_samples.mean(axis=0)  # (batch_size, d_z)
z_std = z_samples.std(axis=0)    # (batch_size, d_z)

print("\n=== Posterior Statistics for z (first 8 patients) ===")
print(f"Mean of z across patients: {z_mean.mean(axis=0)}")
print(f"Std of z across patients: {z_mean.std(axis=0)}")
print(f"\nPosterior uncertainty (avg std dev per dimension): {z_std.mean(axis=0)}")
print(f"Prior std (should be 1.0): {np.ones(model_config.z_dim)}")
print(f"\nRatio of posterior/prior std: {z_std.mean(axis=0) / 1.0}")

print("\n=== Per-patient posterior uncertainty ===")
for i in range(min(3, batch_size)):
    print(f"\nPatient {i} (Response={mini_batch.responses[i].item():.0f}):")
    print(f"  z mean: {z_mean[i]}")
    print(f"  z std:  {z_std[i]}")
    print(f"  Signal-to-noise ratio: {np.abs(z_mean[i]) / z_std[i]}")

# %% Check global parameters
if "W" in samples:
    W_samples = samples["W"].cpu().numpy()  # (num_samples, C, d_z)
    W_mean = W_samples.mean(axis=0)
    W_std = W_samples.std(axis=0)
    print(f"\n=== Global parameter W ===")
    print(f"W mean norm: {np.linalg.norm(W_mean):.3f}")
    print(f"W posterior std (avg): {W_std.mean():.3f}")
    print(f"W prior std: 1.5")
    print(f"Ratio: {W_std.mean() / 1.5:.3f}")

if "T" in samples:
    T_samples = samples["T"].cpu().numpy()  # (num_samples, C, C)
    T_mean = T_samples.mean(axis=0)
    T_std = T_samples.std(axis=0)
    print(f"\n=== Global parameter T ===")
    print(f"T mean norm: {np.linalg.norm(T_mean):.3f}")
    print(f"T posterior std (avg): {T_std.mean():.3f}")
    print(f"T is sampled from Laplace(0, lambda_T)")

if "lambda_T" in samples:
    lambda_T_samples = samples["lambda_T"].cpu().numpy()
    print(f"\n=== lambda_T ===")
    print(f"lambda_T mean: {lambda_T_samples.mean():.3f}")
    print(f"lambda_T std: {lambda_T_samples.std():.3f}")
    print(f"lambda_T prior: Beta(3, 4) with mean = 3/7 ≈ 0.43")
