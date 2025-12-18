# %% Import necessary libraries
import os

import pyro
import matplotlib.pyplot as plt
import torch
import yaml
from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.patient_data import SimulatedDataGenerator
from hbrace.patient_data.dataset import get_train_test_dataloaders

# %% Load configuration with config path
config_path = "configs/experiment.yaml"
run_name, model_config, vi_config, data_config = load_config(config_path)

data_path = f"./data/synthetic_data_{run_name}"
checkpoint_path = f"saved_models/checkpoint_{run_name}.pth"

torch.manual_seed(data_config.seed)
generator = SimulatedDataGenerator.from_model_config(
    model_config=model_config,
    n_patients=data_config.num_patients,
    seed=data_config.seed,
    data_config=data_config,
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

# Save the config to the results/{run_name}/config.yaml
os.makedirs(f"results/{run_name}", exist_ok=True)
with open(f"results/{run_name}/config.yaml", "w") as f:
    yaml.dump(
        {
            "run_name": run_name,
            "model": model_config.__dict__,
            "vi": vi_config.__dict__,
            "data": data_config.__dict__,
        },
        f,
    )

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
        seed=data_config.seed,
        oversample=True,
    )

    training_history = model.train(
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        seed=data_config.seed,
        progress=True,
        grad_clip=5.0,
    )
    train_elbo_history = training_history["train_elbo"]
    val_nll_history = training_history["val_nll"]
    val_elbo_history = training_history.get("val_elbo", [])

    # Visualize the training/validation curves
    os.makedirs(f"results/{run_name}", exist_ok=True)
    plt.plot(train_elbo_history, label="train elbo")
    if val_elbo_history:
        plt.plot(val_elbo_history, label="val elbo")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.title("Training progress (ELBO)")
    plt.legend()
    plt.savefig(f"results/{run_name}/elbo_curve.png")
    plt.savefig(f"results/{run_name}/elbo_curve.svg")

    if val_nll_history:
        plt.figure()
        plt.plot(val_nll_history, label="val nll", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Negative log likelihood")
        plt.title("Validation NLL")
        plt.legend()
        plt.savefig(f"results/{run_name}/val_nll_curve.png")
        plt.savefig(f"results/{run_name}/val_nll_curve.svg")
    model.save_checkpoint(checkpoint_path)
else:
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_checkpoint(checkpoint_path)
