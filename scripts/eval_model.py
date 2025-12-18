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

from hbrace.models.utils import auprc_for_responses, predictive_log_likelihood
from hbrace.models.guides import build_guide
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

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

print("\nPredicted probability statistics:")
print(f"  Min:  {y_score.min():.4f}")
print(f"  Max:  {y_score.max():.4f}")
print(f"  Mean: {y_score.mean():.4f}")
print(f"  Std:  {y_score.std():.4f}")

# %% Find optimal threshold from precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_score)

# Calculate F1 score for each threshold to find optimal
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold from PR curve: {optimal_threshold:.4f}")

# Compute metrics with optimal threshold
y_pred_optimal = (y_score > optimal_threshold).astype(int)

f1_optimal = f1_score(y_true, y_pred_optimal, average='macro')
recall_optimal = recall_score(y_true, y_pred_optimal, average='macro')
precision_optimal = precision_score(y_true, y_pred_optimal, average='macro', zero_division=0)
accuracy_optimal = accuracy_score(y_true, y_pred_optimal)

print(f"\nMetrics with optimal threshold ({optimal_threshold:.4f}):")
print(f"  F1 score: {f1_optimal:.3f}")
print(f"  Recall: {recall_optimal:.3f}")
print(f"  Precision: {precision_optimal:.3f}")
print(f"  Accuracy: {accuracy_optimal:.3f}")

# %% Plot the PR curve with optimal threshold marked
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f"AUPRC={auprc:.3f}", linewidth=2)
plt.scatter(recall[optimal_idx], precision[optimal_idx], color='red', s=100,
            label=f'Optimal (threshold={optimal_threshold:.3f}, F1={f1_scores[optimal_idx]:.3f})', zorder=5)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision-Recall Curve", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"results/{run_name}/pr_curve.png", dpi=150)
plt.savefig(f"results/{run_name}/pr_curve.svg")
print(f"\nSaved PR curve to results/{run_name}/pr_curve.png")