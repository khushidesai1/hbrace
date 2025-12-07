# Hierarchical Breast Cancer Response Analysis (HBRACE)

An application of the hierarchical causal model inspired by the CAIRE formulation in [Wienstein et. al, 2024] for estimating patient response in breast cancer T cells upon receiving treatment of anti-PD1. This method builds on the idea that T cell compositions can heavily shape patient response. We collapse a hierarchical model to a patient level causal graph and use causal inference to analyze the effects of perturbing either the pre-treatment or post-treatment T cell composition of a patient for different subtypes of breast cancer.

