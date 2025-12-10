"""Test semi-synthetic data generation with known ground truth."""
import torch

from hbrace.config import load_config
from hbrace.patient_data.sergio_data import SemiSyntheticDataGenerator

# Load configuration
config_path = "configs/experiment.yaml"
config = load_config(config_path)

# Create semi-synthetic data generator
print("Generating semi-synthetic data with known ground truth...")
generator = SemiSyntheticDataGenerator(
    n_patients=8,
    n_genes=30,
    n_cell_types=5,
    seed=42,
)

# Generate batch with ground truth
batch, ground_truth = generator.generate_batch(config.model, device="cpu")

print(f"\nGenerated batch:")
print(f"  Pre-counts shape: {batch.pre_counts.shape}")
print(f"  On-counts shape: {batch.on_counts.shape}")
print(f"  Responses shape: {batch.responses.shape}")
print(f"  Subtype IDs: {batch.subtype_ids}")

print(f"\nGround truth available:")
for key, value in ground_truth.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: shape {value.shape}")
    else:
        print(f"  {key}: {type(value)}")

print(f"\nPatient responses: {batch.responses.numpy()}")
print(f"Ground truth z (first 3 patients):")
print(ground_truth['z'][:3])
