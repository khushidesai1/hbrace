import numpy as np
from hbrace.config import load_config
from hbrace.patient_data import SimulatedDataGenerator

# %% Load config and synthetic data
run_name, model_config, vi_config, data_config = load_config("configs/experiment.yaml")
sim_data = SimulatedDataGenerator.load(f"./data/synthetic_data_{run_name}")

subtypes = np.asarray(sim_data.subtype_ids)
N, C, G = sim_data.pre_counts.shape

print(f"N={N}, C={C}, G={G}")
print("Subtype counts:", {s: int((subtypes == s).sum()) for s in range(model_config.n_subtypes)})

num_responders = (sim_data.responses == 1).sum()
num_non_responders = (sim_data.responses == 0).sum()
print(f"Number of responders: {num_responders}, Number of non-responders: {num_non_responders}")
print(f"Responder rate: {num_responders / (num_responders + num_non_responders)}")

# %% Mixture variability (pi_p, pi_t)
def summarize_mixtures(name, mat):
    # mat: (N, C)
    per_subtype = []
    for s in range(model_config.n_subtypes):
        mask = subtypes == s
        if not mask.any():
            continue
        sub_mat = mat[mask]
        per_subtype.append(
            {
                "subtype": s,
                "mean": sub_mat.mean(axis=0),
                "std": sub_mat.std(axis=0),
            }
        )
    return per_subtype

print("\nPre-treatment mixtures (pi_p) mean/std per subtype:")
for row in summarize_mixtures("pi_p", sim_data.pi_p):
    print(f"  subtype {row['subtype']}: mean {row['mean']}, std {row['std']}")

print("\nPost-treatment mixtures (pi_t) mean/std per subtype:")
for row in summarize_mixtures("pi_t", sim_data.pi_t):
    print(f"  subtype {row['subtype']}: mean {row['mean']}, std {row['std']}")

# %% Counts variability (pre/on)
pre_counts = sim_data.pre_counts  # (N, C, G)
on_counts = sim_data.on_counts
for label, arr in [("pre", pre_counts), ("post", on_counts)]:
    overall_var = arr.var(axis=0).mean()  # mean variance across C,G
    print(f"\n{label} counts: mean variance across cell types/genes = {overall_var:.3f}")
    for s in range(model_config.n_subtypes):
        mask = subtypes == s
        if not mask.any():
            continue
        sub_var = arr[mask].var(axis=0).mean()
        print(f"  subtype {s}: mean variance = {sub_var:.3f}")

# %% Latent parameter spreads if saved
extra = sim_data.extra_params or {}
for key in ["Delta_std", "Delta", "lambda_T", "W_P", "delta_ic", "z", "u", "beta0", "beta_t", "gamma", "beta_s"]:
    if key in extra:
        arr = np.asarray(extra[key])
        print(f"\n{key}: shape {arr.shape}, mean {arr.mean():.4f}, std {arr.std():.4f}")