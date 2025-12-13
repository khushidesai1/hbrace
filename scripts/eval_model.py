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
from hbrace.models.utils import predictive_log_likelihood, chi_sq_ppc_pvalues_pre_post
from hbrace.models.utils import auprc_for_responses
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

# %% Posterior predictive checks on counts (pre/post) and AUPRC for responses
ppc_samples = 1000
eps = 1e-6

# Chi-squared discrepancy on counts (pre/post)
ppc_counts = chi_sq_ppc_pvalues_pre_post(
    model.model_fn,
    model.guide_fn,
    dataloader_val,
    model_config,
    num_samples=ppc_samples,
    device=torch.device(data_config.device),
    eps=eps,
)
print(
    "\nPosterior predictive p-values (chi-squared on counts): "
    f"pre={ppc_counts['pre']:.3f}, post={ppc_counts['post']:.3f}"
)

# %% AUPRC for response prediction on validation set
auprc, y_true, y_score = auprc_for_responses(
    model.model_fn,
    model.guide_fn,
    dataloader_val,
    num_samples=ppc_samples,
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

# %% Histogram of chi-squared discrepancies on post counts (from ppc_counts runs)
def plot_ppc_hist(run_name, rep, obs, title, outfile):
    plt.figure()
    plt.hist(rep.numpy(), bins=30, color="lightgray", edgecolor="gray")
    plt.axvline(obs.mean().item(), color="black", linestyle="-", linewidth=1.5)
    plt.xlabel(r"$X^2_{\text{min}}(y^{rep})$")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(f"results/{run_name}", exist_ok=True)
    plt.savefig(outfile)

plot_ppc_hist(
    run_name=run_name,
    rep=ppc_counts["pre_rep"],
    obs=ppc_counts["pre_obs"],
    title="PPC chi-squared (pre counts)",
    outfile=f"results/{run_name}/ppc_chi2_pre_counts_hist.png",
)
plot_ppc_hist(
    run_name=run_name,
    rep=ppc_counts["post_rep"],
    obs=ppc_counts["post_obs"],
    title="PPC chi-squared (post counts)",
    outfile=f"results/{run_name}/ppc_chi2_post_counts_hist.png",
)
