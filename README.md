# Robust-Normalization-PINN

An experimental repository for **robust-normalization adaptive weighting** in Physics-Informed Neural Networks (PINNs).  
The core goal of this project is to apply robust statistical normalization (e.g., median/MAD) to PDE residuals and dynamically increase the training weights of "hard samples," thereby improving convergence stability and accuracy.

---

## 1. Project Overview

This repository conducts experiments on three types of PDE problems:

- Allen-Cahn（`AC/`）
- Helmholtz（`Helmholtz/`）
- Poisson（`Poisson/`）

It compares multiple PINN weighting strategies:
- **Vallian PINN**：Normal PINN
- **SA**: Adaptive strategy (Notebook version)
- **RNF**: Robust normalization weighting with fixed hyperparameters (Robust Normalization Fixed)
- **RNA**: Robust normalization weighting with learnable hyperparameters (Robust Normalization Adaptive)

---

## 2. Method Highlights (Robust Normalization)

During training, point-wise residuals are first computed and then robustly standardized:

- Center: residual median
- Scale: MAD (median absolute deviation)
- Map standardized residuals to weights (e.g., with sigmoid)

Intuitively:

- Hard points (larger residuals) receive more attention.
- Easy points (smaller residuals) are suppressed.
- Compared with directly using absolute residual values, robust statistics are more stable to outliers.

---

## 3. Repository Structure

```text
Robust-Normalization-PINN/
├── README.md
├── AC/
│   ├── ac-rna.py                 # AC: RNA (learnable hyperparameters)
│   ├── ac-rnf.py                 # AC: RNF (fixed hyperparameters)
│   ├── ac-sa.ipynb               # AC: SA
│   └── output/
├── Helmholtz/
│   ├── helmholtz-rna.ipynb
│   ├── helmholtz-rnf.ipynb
│   ├── helmholtz-sa.ipynb
│   └── output/
└── Poisson/
	├── possion-rna.ipynb
	├── possion-sa.ipynb
	├── posson-rnf.ipynb
	└── output/
```

> Note: There are spelling inconsistencies (`possion/posson`) in file names under the Poisson directory, which reflect the current naming state of this repository.
