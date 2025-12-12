from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from hbrace.patient_data.types import SimulatedData

class PatientIndexDataset(Dataset):
    """Dataset that returns patient indices to be collated into PatientBatch."""

    def __init__(self, indices: List[int]):
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> int:
        return self.indices[idx]


def get_train_test_dataloaders(
    num_patients: int,
    batch_size: int,
    device: str,
    sim_data: SimulatedData,
    test_fraction: float = 0.25,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test DataLoaders for a given number of patients.

    Args:
        num_patients: Number of patients.
        batch_size: Batch size.
        device: Device to move tensors to.
        sim_data: SimulatedData object containing the patient data.
        test_fraction: Fraction of patients to allocate to the test set.
        seed: Optional seed for deterministic splits and shuffling.

    Returns:
        Tuple of train and test DataLoaders.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    # Get test and train count
    test_count = max(1, int(num_patients * test_fraction))
    test_count = min(test_count, num_patients - 1)
    train_count = num_patients - test_count

    # Get train and test indices
    perm = torch.randperm(num_patients, generator=generator).tolist()
    train_indices = perm[:train_count]
    test_indices = perm[train_count:train_count + test_count]
    
    # Create train and test dataloaders
    train_dataloader = make_dataloader(train_indices, batch_size, device, sim_data, generator)
    test_dataloader = make_dataloader(test_indices, batch_size, device, sim_data, generator) if test_indices else None
    return train_dataloader, test_dataloader


def make_dataloader(
    indices: List[int],
    batch_size: int,
    device: str,
    sim_data: SimulatedData,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """
    Create a DataLoader that returns PatientBatch objects for a list of patient indices.

    Args:
        indices: List of patient indices to include in the DataLoader.
        batch_size: Batch size.
        device: Device to move tensors to.
        sim_data: SimulatedData object containing the patient data.

    Returns:
        DataLoader that returns PatientBatch objects for a list of patient indices.
    """
    return DataLoader(
        PatientIndexDataset(indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        collate_fn=lambda batch_idxs: sim_data.to_patient_batch(
            device=device, indices=batch_idxs
        ),
    )
