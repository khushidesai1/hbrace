from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np
import torch

from hbrace.config import ModelConfig
from hbrace.patient_data.types import PatientBatch, SimulatedData
from hbrace.patient_data.types import SimConfig
from hbrace.patient_data.utils import clr, inv_clr, sample_nb, collapse_cells


class SimulatedDataGenerator:
    """Data generator for fully synthetic data simulation."""

    def __init__(self, sim_config: SimConfig) -> None:
        """
        Initialize the SimulatedDataGenerator.

        Args:
            sim_config: The SimConfig to use for the simulation.
        """
        self.sim_config = sim_config

    @classmethod
    def from_model_config(
        cls,
        model_config: ModelConfig,
        n_patients: int,
        seed: int | None = None,
    ) -> "SimulatedDataGenerator":
        """
        Create a SimulatedDataGenerator from a ModelConfig.

        Args:
            model_config: The ModelConfig to use for the simulation.
            n_patients: The number of patients to simulate.
            seed: The seed to use for the random number generator.
        """
        sim_config = SimConfig(
            n_patients=n_patients,
            n_subtypes=model_config.n_subtypes,
            n_cell_types=model_config.n_cell_types,
            n_genes=model_config.n_genes,
            d_z=model_config.latent_dim,
            r_u=model_config.latent_dim,
            seed=seed if seed is not None else SimConfig.seed,
        )
        return cls(sim_config)

    def generate(self) -> SimulatedData:
        """
        Generate a cohort that mirrors the hierarchical story in the proposal.

        Returns:
            A SimulatedData object containing aggregated patient x cell-type x gene counts with the ground
            truth latent variables stored in "latents" key.
        """

        rng = np.random.default_rng(self.sim_config.seed)
        N, S, C, G, d, r = (
            self.sim_config.n_patients,
            self.sim_config.n_subtypes,
            self.sim_config.n_cell_types,
            self.sim_config.n_genes,
            self.sim_config.d_z,
            self.sim_config.r_u,
        )

        # Subtype assignments and subtype-specific priors over T-cell states.
        subtype_ids = rng.integers(low=0, high=S, size=N)
        theta = np.zeros((S, C))
        for subtype in range(S):
            bump = np.zeros(C)
            bump[subtype % C] = 3.0
            theta[subtype] = rng.dirichlet(2.0 * np.ones(C) + bump)

        # Pre-treatment mixture pi_i^p with tau_i ~ Gamma(2, 0.1).
        tau = rng.gamma(shape=2.0, scale=0.1, size=N)
        pi_p = np.zeros((N, C))
        for i in range(N):
            pi_p[i] = rng.dirichlet(tau[i] * theta[subtype_ids[i]])

        # Base NB parameters for pre-treatment f^p_c(x).
        log_mu_p = rng.normal(loc=1.5, scale=0.8, size=(C, G))
        mu_p = np.exp(log_mu_p)
        phi_p = np.exp(rng.normal(loc=1.0, scale=0.5, size=C))

        # Latent treatment effects and confounders.
        z = rng.normal(loc=0.0, scale=1.0, size=(N, d))
        u = rng.normal(loc=0.0, scale=1.0, size=(N, r))

        # Phenotypic shifts for on-treatment counts.
        Delta = rng.normal(loc=0.0, scale=self.sim_config.sigma_D, size=(C, G, d))
        tau_c = np.abs(rng.normal(loc=0.0, scale=0.5, size=C)) + 1e-3
        delta_ic = rng.normal(loc=0.0, scale=tau_c[None, :], size=(N, C))

        # Composition shift eta^t_i = eta^p_i + T W_P z_i + epsilon_i.
        adjacency = rng.random((C, C)) < 0.3
        np.fill_diagonal(adjacency, 0)
        lambda_h = np.abs(rng.normal(loc=0.0, scale=0.2))
        lambda_l = np.abs(rng.normal(loc=0.0, scale=0.05))
        scale_T = np.where(adjacency, lambda_l, lambda_h)
        T = rng.choice([-1.0, 1.0], size=(C, C)) * rng.exponential(scale=scale_T)
        W_P = rng.normal(loc=0.0, scale=self.sim_config.sigma_W, size=(C, d))

        eta_p = clr(pi_p)
        eps = rng.normal(loc=0.0, scale=self.sim_config.sigma_eps, size=(N, C))
        eta_t = eta_p + (z @ (T @ W_P).T) + eps
        pi_t = inv_clr(eta_t)

        # On-treatment NB params mu^t_icg, phi^t_ic.
        dot_Dz = np.tensordot(z, Delta, axes=([1], [2]))  # (N, C, G)
        mu_t = np.exp(log_mu_p[None, :, :] + dot_Dz)
        phi_t = phi_p[None, :] * np.exp(delta_ic)

        # Sample cell-level data and track cell types.
        pre_cells: List[np.ndarray] = []
        pre_cell_types: List[np.ndarray] = []
        post_cells: List[np.ndarray] = []
        post_cell_types: List[np.ndarray] = []
        for i in range(N):
            ct_pre = rng.choice(C, size=self.sim_config.m_pre, p=pi_p[i])
            X_pre = np.zeros((self.sim_config.m_pre, G), dtype=np.int64)
            for j, c_idx in enumerate(ct_pre):
                X_pre[j] = sample_nb(mu_p[c_idx], np.repeat(phi_p[c_idx], G), rng)
            pre_cells.append(X_pre)
            pre_cell_types.append(ct_pre)

            ct_post = rng.choice(C, size=self.sim_config.n_post, p=pi_t[i])
            X_post = np.zeros((self.sim_config.n_post, G), dtype=np.int64)
            for k, c_idx in enumerate(ct_post):
                X_post[k] = sample_nb(mu_t[i, c_idx], np.repeat(phi_t[i, c_idx], G), rng)
            post_cells.append(X_post)
            post_cell_types.append(ct_post)

        pre_counts = collapse_cells(pre_cells, pre_cell_types, C)
        on_counts = collapse_cells(post_cells, post_cell_types, C)

        # Patient response y_i via logistic regression on composition, u, and subtype.
        beta0 = rng.normal(0.0, 0.5)
        beta_t = rng.normal(0.0, 0.5, size=C)
        gamma = rng.normal(0.0, 0.5, size=r)
        beta_s = rng.normal(0.0, 0.5, size=S)
        linear = (
            beta0
            + (pi_t * beta_t[None, :]).sum(axis=1)
            + (u * gamma[None, :]).sum(axis=1)
            + beta_s[subtype_ids]
        )
        prob = 1.0 / (1.0 + np.exp(-linear))
        responses = rng.binomial(1, prob, size=N)

        latents: Dict[str, Any] = {
            "z": z,
            "u": u,
            "mu_p": mu_p,
            "phi_p": phi_p,
            "mu_t": mu_t,
            "phi_t": phi_t,
            "theta": theta,
            "tau": tau,
            "T": T,
            "W_P": W_P,
            "Delta": Delta,
            "delta_ic": delta_ic,
            "beta0": beta0,
            "beta_t": beta_t,
            "gamma": gamma,
            "beta_s": beta_s,
            "pre_cell_types": pre_cell_types,
            "post_cell_types": post_cell_types,
        }

        return SimulatedData(
            config=self.sim_config,
            pre_counts=pre_counts,
            on_counts=on_counts,
            responses=responses,
            subtype_ids=subtype_ids,
            pi_p=pi_p,
            pi_t=pi_t,
            latents=latents,
        )


    def generate_batch(
        self,
        device: torch.device | str = "cpu",
        return_simulation: bool = False,
        save: bool = False,
        out_dir: str | Path = "./data",
        name: str | None = None,
    ) -> PatientBatch | Tuple[PatientBatch, SimulatedData]:
        """
        Generate and immediately convert to a PatientBatch.

        Args:
            device: Device to move tensors to.
            return_simulation: Whether to return the full SimulatedData object.
            save: If True, persist generated arrays to disk (uncompressed .npy).
            out_dir: Directory to place saved files (created if missing).
            name: Optional stem for saved files (defaults to "sim_data").

        Returns:
            A PatientBatch object on the specified device. If return_simulation is True,
            returns a tuple of (PatientBatch, SimulatedData).
        """

        sim_data = self.generate()
        if save:
            self._save(sim_data, out_dir=Path(out_dir), name=name)
        batch = sim_data.to_patient_batch(device=device)
        if return_simulation:
            return batch, sim_data
        return batch

    @staticmethod
    def _save(sim_data: SimulatedData, out_dir: Path, name: str | None = None) -> None:
        """Persist key arrays and config to disk using plain .npy and .json."""

        stem = name or "sim_data"
        (out_dir / stem).mkdir(parents=True, exist_ok=True)
        np.save(out_dir / stem / f"{stem}_pre_counts.npy", sim_data.pre_counts)
        np.save(out_dir / stem / f"{stem}_on_counts.npy", sim_data.on_counts)
        np.save(out_dir / stem / f"{stem}_responses.npy", sim_data.responses)
        np.save(out_dir / stem / f"{stem}_subtype_ids.npy", sim_data.subtype_ids)
        np.save(out_dir / stem / f"{stem}_pi_p.npy", sim_data.pi_p)
        np.save(out_dir / stem / f"{stem}_pi_t.npy", sim_data.pi_t)

        cfg = sim_data.config.__dict__
        (out_dir / stem / f"{stem}_config.json").write_text(json.dumps(cfg, indent=2))

    @staticmethod
    def load(path_stem: str | Path) -> SimulatedData:
        """
        Rehydrate a SimulatedData object from saved .npy/.json files.

        Args:
            path_stem: Base path without the suffix (e.g., './data/sim_data').
                       The loader will look for '<stem>_*.npy' and '<stem>_config.json'.
        """

        stem = Path(path_stem)
        cfg_path = stem / f"{stem.name}_config.json"
        config_dict = json.loads(cfg_path.read_text())
        sim_config = SimConfig(**config_dict)

        pre_counts = np.load(stem / f"{stem.name}_pre_counts.npy")
        on_counts = np.load(stem / f"{stem.name}_on_counts.npy")
        responses = np.load(stem / f"{stem.name}_responses.npy")
        subtype_ids = np.load(stem / f"{stem.name}_subtype_ids.npy")
        pi_p = np.load(stem / f"{stem.name}_pi_p.npy")
        pi_t = np.load(stem / f"{stem.name}_pi_t.npy")

        return SimulatedData(
            config=sim_config,
            pre_counts=pre_counts,
            on_counts=on_counts,
            responses=responses,
            subtype_ids=subtype_ids,
            pi_p=pi_p,
            pi_t=pi_t,
            latents={},
        )
