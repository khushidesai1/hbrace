# Data Directory

This directory contains the real immunotherapy scRNA-seq data for model training and validation.

## Files

- `wu_cohort1.h5ad`: Real patient data from Bassez et al. 2021 breast cancer immunotherapy study
  - Treatment-naive cohort (anti-PD1 only)
  - Contains annotated T cells with clonal expansion information
  - Pre-treatment and on-treatment samples
  - Size: ~2.3GB

## Source

Data from: Bassez, A., Vos, H., Van Dyck, L. et al. A single-cell map of intratumoral changes during anti-PD1 treatment of patients with breast cancer. *Nature Medicine* 27, 820–832 (2021). https://doi.org/10.1038/s41591-021-01323-8

## Note

Data files are intentionally excluded from git (see `.gitignore`). To use this project, you'll need to download the data separately.
