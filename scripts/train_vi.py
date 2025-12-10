# %% Import necessary libraries
import os
import torch

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.patient_data import SimulatedDataGenerator

# %% Load configuration with config path
data_path = "./data/synthetic_data_test"
config_path = "configs/experiment.yaml"
num_patients = 8
device = "cpu"
seed = 42

model_config, vi_config = load_config(config_path)
torch.manual_seed(seed)
generator = SimulatedDataGenerator.from_model_config(
    model_config=model_config,
    n_patients=num_patients,
    seed=seed,
)
if not os.path.exists(data_path):
    print(f"Generating data and saving to {data_path}")
    batch, _ = generator.generate_batch(
        device=device, 
        return_simulation=True, 
        save=True, 
        name=data_path.split("/")[-1]
    )
else:
    print(f"Loading data from {data_path}")
    sim_data = SimulatedDataGenerator.load(data_path)
    batch = sim_data.to_patient_batch(device=device)

# %% Initialize and train the model
model = HBRACEModel(model_config, vi_config)
model.train(batch, seed=seed)
