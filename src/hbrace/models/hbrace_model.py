from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from hbrace.config import ModelConfig, VIConfig
from hbrace.patient_data import PatientBatch
from .guides import build_guide
from .hierarchical import hierarchical_model
from .utils import EarlyStopping, predictive_log_likelihood

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover - fallback if tqdm is missing
    trange = None


class HBRACEModel:
    """HBRACE model wrapper around the hierarchical Pyro model."""

    def __init__(self, model_config: ModelConfig, vi_config: VIConfig) -> None:
        """
        Initialize the HBRACEModel.

        Args:
            model_config: The ModelConfig to use for the model.
            vi_config: The VIConfig to use for the model.
        """
        self.model_config = model_config
        self.vi_config = vi_config
        self.model_fn = functools.partial(hierarchical_model, config=model_config)
        self.early_stopping = EarlyStopping(patience=float(vi_config.early_stopping_patience))

    def train(
        self,
        dataloader_train: Iterable[PatientBatch],
        dataloader_val: Optional[Iterable[PatientBatch]] = None,
        seed: int = 0,
        progress: bool = True,
        val_samples: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Run stochastic variational inference across epochs of patient batches.

        Args:
            dataloader_train: Iterable yielding PatientBatch objects for training.
            dataloader_val: Iterable yielding validation PatientBatch objects.
            seed: RNG seed for reproducibility.
            progress: Whether to show a tqdm progress bar when available.
            val_samples: Number of posterior samples to use for predictive log likelihood.
        """
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)

        self.guide_fn = build_guide(
            self.model_fn,
            self.model_config,
            self.vi_config.guide,
            rank=self.vi_config.guide_rank,
        )
        optimizer = ClippedAdam({"lr": self.vi_config.learning_rate})
        svi = SVI(self.model_fn, self.guide_fn, optimizer, loss=Trace_ELBO())

        use_tqdm = progress and trange is not None
        iterator = trange(self.vi_config.num_epochs) if use_tqdm else range(self.vi_config.num_epochs)

        train_history: List[float] = []
        val_history: List[float] = []
        last_train_elbo = float("nan")

        for epoch in iterator:
            train_loss_total = 0.0
            train_n_obs = 0

            for batch in dataloader_train:
                loss = svi.step(batch)
                train_loss_total += loss
                train_n_obs += batch.responses.shape[0]
            train_elbo = train_loss_total / max(train_n_obs, 1)
            train_history.append(train_elbo)

            val_nll = float("nan")
            if dataloader_val is not None:
                val_nll = -predictive_log_likelihood(
                    self.model_fn, self.guide_fn, dataloader_val, num_samples=val_samples
                )
                val_history.append(val_nll)

            if use_tqdm:
                iterator.set_description(
                    f"Epoch {epoch} | train elbo: {train_elbo:.2f} (last: {last_train_elbo:.2f}) | val nll: {val_nll:.2f}"
                )
            elif self.vi_config.log_interval and epoch % self.vi_config.log_interval == 0:
                print(
                    f"Epoch {epoch} | train elbo: {train_elbo:.2f} (last: {last_train_elbo:.2f}) | val nll: {val_nll:.2f}"
                )
            last_train_elbo = train_elbo
            if dataloader_val is not None and self.early_stopping(val_nll):
                break
        
        if self.early_stopping.has_stopped():
            print("Early stopping triggered")
            print(f"Best val nll: {self.early_stopping.validation_loss_min:.2f}")
            print(f"Best epoch: {epoch}")

        self.history = {"train_elbo": train_history, "val_nll": val_history}
        return self.history

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist the current Pyro parameter store to disk."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pyro.get_param_store().save(str(path))

    def load_checkpoint(self, path: str | Path, map_location: Optional[str] = None) -> None:
        """
        Load Pyro parameters from disk. Call after constructing the model/guide.

        Args:
            path: File produced by save_checkpoint.
            map_location: Optional device mapping for tensors (e.g., 'cpu').
        """
        pyro.get_param_store().load(str(Path(path)), map_location=map_location)
