# %% Import necessary libraries
import os

import pyro
import matplotlib.pyplot as plt
import torch

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.patient_data import SimulatedDataGenerator
from hbrace.patient_data.dataset import get_train_test_dataloaders

# %% Load configuration with config path
data_path = "./data/synthetic_data_test"
config_path = "configs/experiment.yaml"
checkpoint_path = "saved_models/checkpoint.pth"

model_config, vi_config, data_config = load_config(config_path)

torch.manual_seed(data_config.seed)
generator = SimulatedDataGenerator.from_model_config(
    model_config=model_config,
    n_patients=data_config.num_patients,
    seed=data_config.seed,
)
if not os.path.exists(data_path):
    print(f"Generating data and saving to {data_path}")
    _, sim_data = generator.generate_batch(
        device=data_config.device, 
        return_simulation=True, 
        save=True, 
        name=data_path.split("/")[-1]
    )
else:
    print(f"Loading data from {data_path}")
    sim_data = SimulatedDataGenerator.load(data_path)

# %% Create train/test split at the patient level
model = HBRACEModel(model_config, vi_config)

if not os.path.exists(checkpoint_path):
    num_patients = sim_data.pre_counts.shape[0]
    G = model_config.n_genes

    dataloader_train, dataloader_val = get_train_test_dataloaders(
        num_patients=num_patients,
        batch_size=data_config.batch_size,
        device=data_config.device,
        sim_data=sim_data,
        test_fraction=data_config.test_fraction,
    )

    training_history = model.train(
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        seed=data_config.seed,
        progress=True,
    )
    train_elbo_history = training_history["train_elbo"]
    val_nll_history = training_history["val_nll"]

    # Visualize the training/validation curves
    os.makedirs("results", exist_ok=True)
    plt.plot(train_elbo_history, label="train elbo")
    if val_nll_history:
        plt.plot(val_nll_history, label="val nll")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training progress")
    plt.legend()
    plt.savefig("results/elbo_curve.png")

    model.save_checkpoint(checkpoint_path)
else:
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_checkpoint(checkpoint_path)

# %% Visualize some of the learned parameters (T, z, u)

# %% Evaluate posterior predictive on held-out (or training if none)
