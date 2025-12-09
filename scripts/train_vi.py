# %% Import necessary libraries
import torch

from hbrace.config import load_config
from hbrace.patient_data import sample_synthetic_batch
from hbrace.models import HBRACEModel

# %% Load configuration with config path
config_path = "configs/experiment.yaml"
num_patients = 8
device = "cpu"
seed = 42

config = load_config(config_path)
torch.manual_seed(seed)
batch = sample_synthetic_batch(
    model_config=config.model,
    n_patients=num_patients,
    device=device,
    seed=seed,
)

# %% Initialize and train the model
model = HBRACEModel(config)
model.train(batch, seed=seed)
