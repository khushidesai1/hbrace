import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from hbrace.config import load_config
from hbrace.patient_data import SimulatedDataGenerator

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# %% Load config and synthetic data
run_name, model_config, vi_config, data_config = load_config(
    "results/model_1000_genes/config.yaml"
)
sim_data = SimulatedDataGenerator.load(f"./data/synthetic_data_{run_name}")

subtypes = np.asarray(sim_data.subtype_ids)
N, C, G = sim_data.pre_counts.shape

print(f"N={N}, C={C}, G={G}")
print("Subtype counts:", {s: int((subtypes == s).sum()) for s in range(model_config.n_subtypes)})

num_responders = (sim_data.responses == 1).sum()
num_non_responders = (sim_data.responses == 0).sum()
print(f"Number of responders: {num_responders}, Number of non-responders: {num_non_responders}")
print(f"Responder rate: {num_responders / (num_responders + num_non_responders):.4f}")

# Create output directory
os.makedirs(f"results/{run_name}", exist_ok=True)

# %% Mixture variability (pi_p, pi_t)
def summarize_mixtures(mat):
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
for row in summarize_mixtures(sim_data.pi_p):
    print(f"  subtype {row['subtype']}: mean {row['mean']}, std {row['std']}")

print("\nPost-treatment mixtures (pi_t) mean/std per subtype:")
for row in summarize_mixtures(sim_data.pi_t):
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
for key in [
    "Delta_std",
    "Delta",
    "lambda_T",
    "W_P",
    "V",
    "delta_ic",
    "z",
    "u",
    "beta0",
    "beta_t",
    "gamma",
    "beta_s",
]:
    if key in extra:
        arr = np.asarray(extra[key])
        if arr is not None:
            print(f"\n{key}: shape {arr.shape}, mean {arr.mean():.4f}, std {arr.std():.4f}")
        else:
            print(f"\n{key}: None (not used in this model)")

# %% Stacked barplot: pre vs post compositions for all subtypes
print("Generating stacked composition barplot...")

n_subtypes = model_config.n_subtypes
cell_type_names = [f"CT{i}" for i in range(C)]

# Compute mean compositions per subtype
mean_pi_p_by_subtype = np.full((n_subtypes, C), np.nan, dtype=float)
mean_pi_t_by_subtype = np.full((n_subtypes, C), np.nan, dtype=float)
n_by_subtype = np.zeros(n_subtypes, dtype=int)

for s in range(n_subtypes):
    mask = subtypes == s
    n_s = int(mask.sum())
    n_by_subtype[s] = n_s
    if n_s == 0:
        continue
    mean_pi_p_by_subtype[s] = sim_data.pi_p[mask].mean(axis=0)
    mean_pi_t_by_subtype[s] = sim_data.pi_t[mask].mean(axis=0)

valid_subtypes = [s for s in range(n_subtypes) if n_by_subtype[s] > 0]
k = len(valid_subtypes)
if k == 0:
    raise ValueError("No subtypes have any patients; cannot plot barplot.")

# Create figure
fig, ax = plt.subplots(figsize=(max(8, k * 1.5), 6))

# X positions for each (subtype, treatment) pair
x_labels = []
x_positions = []
for idx, s in enumerate(valid_subtypes):
    x_labels.append(f"Subtype {s}\nPre (n={n_by_subtype[s]})")
    x_labels.append(f"Subtype {s}\nOn (n={n_by_subtype[s]})")
    x_positions.extend([idx * 2, idx * 2 + 1])

x_positions = np.array(x_positions)

# Color palette for cell types
colors = plt.cm.Set3(np.linspace(0, 1, C))

# Stack cell type proportions
bottoms = np.zeros(len(x_positions))
for c in range(C):
    heights = []
    for s in valid_subtypes:
        heights.append(mean_pi_p_by_subtype[s, c])
        heights.append(mean_pi_t_by_subtype[s, c])

    ax.bar(
        x_positions,
        heights,
        bottom=bottoms,
        width=0.8,
        label=cell_type_names[c],
        color=colors[c],
        edgecolor='white',
        linewidth=0.5
    )
    bottoms += heights

ax.set_xlabel("Subtype and Treatment", fontsize=12, fontweight='bold')
ax.set_ylabel("Cell Type Proportion", fontsize=12, fontweight='bold')
ax.set_title("Cell Type Composition by Subtype (Pre vs On-treatment)", fontsize=14, fontweight="bold")
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis="y")
ax.legend(title="Cell Type", fontsize=9, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

plt.tight_layout()
plt.savefig(
    f"results/{run_name}/composition_stacked_barplot.png",
    dpi=150,
    bbox_inches="tight",
)
plt.savefig(
    f"results/{run_name}/composition_stacked_barplot.svg",
    bbox_inches="tight",
)
print(f"Saved: results/{run_name}/composition_stacked_barplot.png")
plt.close()

print("Dataset summary complete!")
