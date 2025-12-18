import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from hbrace.config import load_config
from hbrace.patient_data import SimulatedDataGenerator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# %% Load config and synthetic data
run_name, model_config, vi_config, data_config = load_config("results/best_model_even_bigger_shift/config.yaml")
sim_data = SimulatedDataGenerator.load(f"./data/synthetic_data_{run_name}")

subtypes = np.asarray(sim_data.subtype_ids)
N, C, G = sim_data.pre_counts.shape

print(f"N={N}, C={C}, G={G}")
print("Subtype counts:", {s: int((subtypes == s).sum()) for s in range(model_config.n_subtypes)})

num_responders = (sim_data.responses == 1).sum()
num_non_responders = (sim_data.responses == 0).sum()
print(f"Number of responders: {num_responders}, Number of non-responders: {num_non_responders}")
print(f"Responder rate: {num_responders / (num_responders + num_non_responders)}")

# Create output directory
os.makedirs(f"results/{run_name}", exist_ok=True)

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
for key in ["Delta_std", "Delta", "lambda_T", "W_P", "V", "delta_ic", "z", "u", "beta0", "beta_t", "gamma", "beta_s"]:
    if key in extra:
        arr = np.asarray(extra[key])
        if arr is not None:
            print(f"\n{key}: shape {arr.shape}, mean {arr.mean():.4f}, std {arr.std():.4f}")
        else:
            print(f"\n{key}: None (not used in this model)")

# %% Visualize composition distributions by subtype
print("Generating composition distribution plots...")

# Create figure with subplots for each subtype
n_subtypes = model_config.n_subtypes
fig, axes = plt.subplots(n_subtypes, 2, figsize=(14, 4*n_subtypes))
if n_subtypes == 1:
    axes = axes.reshape(1, -1)

cell_type_names = [f"CT{i}" for i in range(C)]

for s in range(n_subtypes):
    mask = subtypes == s
    n_patients_subtype = mask.sum()

    if n_patients_subtype == 0:
        continue

    # Pre-treatment compositions (pi_p)
    ax_pre = axes[s, 0]
    pi_p_subtype = sim_data.pi_p[mask]  # (n_patients_subtype, C)

    # Box plot for pre-treatment
    positions = np.arange(C)
    bp_pre = ax_pre.boxplot([pi_p_subtype[:, c] for c in range(C)],
                             positions=positions,
                             widths=0.6,
                             patch_artist=True,
                             showmeans=True,
                             meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    for patch in bp_pre['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax_pre.set_xlabel('Cell Type', fontsize=12)
    ax_pre.set_ylabel('Proportion', fontsize=12)
    ax_pre.set_title(f'Pre-treatment (π_p) - Subtype {s} (n={n_patients_subtype})', fontsize=14, fontweight='bold')
    ax_pre.set_xticks(positions)
    ax_pre.set_xticklabels(cell_type_names, rotation=45)
    ax_pre.set_ylim([0, 1])
    ax_pre.grid(True, alpha=0.3, axis='y')

    # On-treatment compositions (pi_t)
    ax_post = axes[s, 1]
    pi_t_subtype = sim_data.pi_t[mask]  # (n_patients_subtype, C)

    # Box plot for on-treatment
    bp_post = ax_post.boxplot([pi_t_subtype[:, c] for c in range(C)],
                               positions=positions,
                               widths=0.6,
                               patch_artist=True,
                               showmeans=True,
                               meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    for patch in bp_post['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)

    ax_post.set_xlabel('Cell Type', fontsize=12)
    ax_post.set_ylabel('Proportion', fontsize=12)
    ax_post.set_title(f'On-treatment (π_t) - Subtype {s} (n={n_patients_subtype})', fontsize=14, fontweight='bold')
    ax_post.set_xticks(positions)
    ax_post.set_xticklabels(cell_type_names, rotation=45)
    ax_post.set_ylim([0, 1])
    ax_post.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"results/{run_name}/composition_distributions_by_subtype.png", dpi=150, bbox_inches='tight')
plt.savefig(f"results/{run_name}/composition_distributions_by_subtype.pdf", bbox_inches='tight')
print(f"\nSaved: results/{run_name}/composition_distributions_by_subtype.png")
plt.close()

# %% Create a comparison plot showing pre vs post changes by subtype
fig, axes = plt.subplots(1, n_subtypes, figsize=(6*n_subtypes, 5))
if n_subtypes == 1:
    axes = [axes]

for s in range(n_subtypes):
    mask = subtypes == s
    n_patients_subtype = mask.sum()

    if n_patients_subtype == 0:
        continue

    ax = axes[s]

    # Compute mean compositions
    mean_pi_p = sim_data.pi_p[mask].mean(axis=0)
    mean_pi_t = sim_data.pi_t[mask].mean(axis=0)

    x = np.arange(C)
    width = 0.35

    bars1 = ax.bar(x - width/2, mean_pi_p, width, label='Pre-treatment (π_p)',
                   color='lightblue', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, mean_pi_t, width, label='On-treatment (π_t)',
                   color='lightcoral', edgecolor='black', alpha=0.8)

    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Mean Proportion', fontsize=12)
    ax.set_title(f'Subtype {s} Composition Change\n(n={n_patients_subtype})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cell_type_names, rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(mean_pi_p.max(), mean_pi_t.max()) * 1.2])

plt.tight_layout()
plt.savefig(f"results/{run_name}/composition_comparison_by_subtype.png", dpi=150, bbox_inches='tight')
plt.savefig(f"results/{run_name}/composition_comparison_by_subtype.pdf", bbox_inches='tight')
print(f"Saved: results/{run_name}/composition_comparison_by_subtype.png")
plt.close()

plt.tight_layout()
plt.savefig(f"results/{run_name}/composition_heatmap_by_subtype.png", dpi=150, bbox_inches='tight')
plt.savefig(f"results/{run_name}/composition_heatmap_by_subtype.pdf", bbox_inches='tight')
print(f"Saved: results/{run_name}/composition_heatmap_by_subtype.png")
plt.close()

print("Dataset summary complete!")