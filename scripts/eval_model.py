# %% Import necessary libraries
import pyro

from hbrace.config import load_config
from hbrace.models import HBRACEModel

# %% Load configuration with config path
data_path = "./data/synthetic_data_test"
config_path = "configs/experiment.yaml"
checkpoint_path = "saved_models/checkpoint.pth"

model_config, vi_config, data_config = load_config(config_path)

model = HBRACEModel(model_config, vi_config)
model.load_checkpoint(checkpoint_path)

# %% Evaluate posterior predictive on held-out (or training if none)
