# Empirical Audit: Using Adversarial Training to Improve Uncertainty Quantification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![IEEE TAI Audit](https://img.shields.io/badge/Audit-IEEE_TAI_2026-brightgreen.svg)](https://ieeexplore.ieee.org/document/10902522)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete reproduction code, empirical audit suite, and multi-dataset evaluation results for the paper:

> **"Using Adversarial Training to Improve Uncertainty Quantification"**  
> *Kuan Huang, Meng Xu, and Yingfeng Wang*  
> **IEEE Transactions on Artificial Intelligence (TAI)**, Vol. 7, No. 1, January 2026.

---

## 📌 Executive Summary & Key Findings

The original paper proposes that standard neural network Uncertainty Quantification (UQ) fails under adversarial attacks due to *imperfect training*, and claims that **adversarial training consistently improves both classification accuracy and uncertainty-based misclassification detection** across MNIST, CIFAR-10, and SVHN.

### The Core Audit Discovery: The "Static Evaluation" Illusion
Our independent empirical audit reproduces the paper's reported numbers under their exact testing protocol, but reveals that **the reported benefits collapse under realistic threat models**:

| Evaluation Protocol | What We Test | Accuracy Gain ($\Delta\text{Acc}$) | UQ Gain ($\Delta\text{AUC}$) | Reality |
| :--- | :--- | :---: | :---: | :--- |
| **Static Attack** *(Paper's Protocol)* | Pre-computed attack perturbations generated from the *baseline* model. | **$+45.93\%$** ($p < 0.0001$) | **$+0.064$** | **Apparent Success:** The model memorizes the specific training perturbations. |
| **Fresh Attack** *(Realistic Hacker)* | Active, white-box attacks computed directly against the *retrained* model. | **$+0.25\%$** ($p = 0.090$) | **$-0.097$** (95% CI: $[-0.139, -0.057]$) | **Complete Collapse:** The model remains vulnerable, and UQ is *worse* than random guessing ($<0.50$) in 62.2% of cases. |

```
   STATIC PROTOCOL (Paper's Claim)             FRESH PROTOCOL (Realistic Audit)
   Baseline Model -> Attack Perturbation       Retrained Model -> Fresh Attack
         │                                            │
         ▼                                            ▼
   Tested on Retrained Model                   Tested on Retrained Model
         │                                            │
         ▼                                            ▼
   Looks Robust (+45.9% Acc, +0.064 AUC)       Complete Breakdown (+0.25% Acc, -0.097 AUC)
```

---

## 🏆 Multi-Pillar Trust Scorecard

| Evaluation Pillar | Target Standard | Our Audit Finding | Status |
| :--- | :--- | :---: | :---: |
| **Robustness** | Mean $\Delta\text{AUC} > 0$ under fresh white-box attack | **$-0.097$** (95% CI: $[-0.139, -0.057]$) | ❌ **FAIL** |
| **Reliability** | Mean $\text{AUC} \ge 0.70$ and 0 units $< 0.50$ | Mean $0.458$; **239 / 384 units $< 0.50$** | ❌ **FAIL** |
| **Calibration** | Clean ECE $\le 0.08$ and Adversarial ECE $\le 0.15$ | Clean $0.058$, **Adversarial $0.469$** | ❌ **FAIL** |
| **Safety** | Silent-Failure Rate (SFR) $\le 35\%$ at 95% retention | **$90.3\%$ silent failure rate** | ❌ **FAIL** |
| **Reproducibility** | Deterministic seeds, config hashes, and public scripts | Hash `4afe4784`, Seed 0 verified | ✅ **PASS** |

---

## 🖼️ Attack Visualizations & Signatures

This repository includes scripts to visualize what each adversarial attack does to actual images across all three datasets:

### 1. Attack Grid Across Datasets (`figures/presentation_attack_grid.png`)
Shows original clean images vs. the 7 attack variants (`FGSM`, `BIM-A`, `BIM-B`, `APGD-reduced`, `PGDL2`, `Jitter`, `Adaptive`) alongside predicted classes and model confidence.
* Human eyes easily recognize the digit or object.
* Neural networks flip to incorrect predictions with **$95\%-99\%$ inverted certainty**.

### 2. Magnified Noise Signatures (`figures/presentation_attack_signatures.png`)
Reveals the $6\times$ magnified mathematical perturbation ($|\eta| \times 6$) heatmap for each attack:
* **FGSM**: Coarse, uniform pixel-grid salt-and-pepper noise.
* **BIM-A**: Minimal, edge-focused early-stopping perturbations.
* **BIM-B**: Fully saturated limit-cycle perturbations.
* **APGD**: Multi-frequency adaptive gradient patterns.
* **PGDL2**: Smooth, localized Euclidean energy clouds.
* **Adaptive**: Specific boundary perturbations engineered to maximize misclassification while penalizing predictive entropy (inducing silent failures).

---

## 📂 Repository Structure

```text
├── run_audit.py                      # Master multi-dataset audit script (MNIST, CIFAR-10, SVHN)
├── generate_presentation_figures.py  # Generates high-res visual attack grids & noise heatmaps
├── TAIML_AdvUQ_Colab.ipynb           # Interactive reproduction notebook for Google Colab (T4 GPU)
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Clean repository tracking (excludes raw data and caches)
│
├── results/                          # Audit deliverables and raw logs
│   ├── TAIML_Report_Results_MultiDataset.docx  # Full written audit report (Sections 8-13)
│   ├── all_results_4afe4784.csv                # Complete 2,568 evaluation rows across all units
│   └── config_4afe4784.json                    # Exact audit configuration parameters
│
└── figures/                          # Publication-grade evaluation figures
    ├── c1_heatmap.png                          # Baseline UQ degradation heatmap
    ├── c3_static_fresh.png                     # Static vs. Fresh robustness gain comparison
    ├── presentation_attack_grid.png            # Visual attack grid on MNIST, CIFAR, SVHN
    └── presentation_attack_signatures.png      # 6x magnified perturbation heatmaps
```

> **Note on Datasets:** In accordance with standard submission practices, large raw dataset binaries (`data/`) and temporary model cache files (`cache/`) are excluded from Git tracking via `.gitignore`. The audit script automatically downloads or formats standard torchvision datasets when run.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the requirements:
```bash
git clone https://github.com/dhruvv1101/TAIML-AdvUQ-Audit.git
cd TAIML-AdvUQ-Audit
pip install -r requirements.txt
```

### 2. Run the Full Audit
To run the complete audit pipeline across MNIST, CIFAR-10, and SVHN:
```bash
python run_audit.py
```
This executes:
1. Balanced dataset loading and Exploratory Data Analysis (EDA).
2. MC Dropout and Bayesian Neural Network (BNN) baseline training.
3. Adversarial retraining across `BIM-A`, `BIM-B`, `APGD-reduced`, and `Adaptive`.
4. Side-by-side evaluation of both **Static** and **Fresh** re-attack protocols.
5. Compilation of `all_results_*.csv` and `TAIML_Report_Results_MultiDataset.docx`.

### 3. Generate Presentation Visuals
To generate the visual attack comparison grid and magnified noise signature heatmaps:
```bash
python generate_presentation_figures.py
```
Outputs are saved directly to `figures/`.

### 4. Interactive Google Colab Execution
Open `TAIML_AdvUQ_Colab.ipynb` directly in Google Colab. Set runtime to **GPU (T4)** and select **Runtime > Run All**.

---

## 📊 Summary of Claims & Verdicts

* **C1: Baseline UQ Breakdown (Supported):** Baseline models suffer catastrophic failure under attack; accuracy drops to $3.3\% - 15.1\%$ and silent-failure rate reaches $90.3\%$.
* **C2: MC Dropout vs. BNN Equivalence (Supported):** MC Dropout and BNN achieve comparable accuracy and UQ performance; variational approximation is not the cause of UQ failure.
* **C3: Static vs. Fresh Robustness (Refuted / Clarified):** Adversarial training improves UQ under static evaluations ($+0.064$ AUC), but completely collapses under fresh white-box re-attacks ($-0.097$ AUC).
* **C4: Accuracy vs. UQ Gain Disconnect (Supported):** Accuracy improvements from adversarial training do not correlate with UQ improvements (Spearman $\rho \approx 0.02, p > 0.05$).
* **C5: Epsilon Sensitivity Sweep (Supported):** Robustness does not monotonically scale with perturbation radius $\epsilon$.
* **C6: BUSI / Complex Transfer (Supported):** Similar patterns hold across complex imaging domains.

---

## 📄 License & Attribution

This audit project is released under the **MIT License**.  
For academic citations or references, please cite the original IEEE TAI 2026 paper alongside this audit repository.
