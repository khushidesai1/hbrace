# %% Import necessary libraries
import torch

from hbrace.config import load_config
from hbrace.models import HBRACEModel
from hbrace.patient_data import SimulatedDataGenerator

# %% Load configuration with config path
config_path = "configs/experiment.yaml"
num_patients = 8
device = "cpu"
seed = 42

config = load_config(config_path)
torch.manual_seed(seed)
generator = SimulatedDataGenerator.from_model_config(
    model_config=config.model,
    n_patients=num_patients,
    seed=seed,
)
batch, _ = generator.generate_batch(device=device, return_simulation=True)

# %% Initialize and train the model
model = HBRACEModel(config)
model.train(batch, seed=seed)
