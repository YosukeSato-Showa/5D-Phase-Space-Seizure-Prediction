# 5D Phase Space Transition Tracking for Precision Seizure Prediction

This repository contains the complete dataset and computational codebase to reproduce the findings of our study on personalized seizure prediction, specifically targeting a deterministic preictal phase transition using a Subject-Conditioned Conformer architecture.

**Target Journal:** *Nature Communications*

## Overview
Existing computational models for seizure forecasting suffer from a profound performance drop when applied to unseen patients due to inter-subject phenotypic heterogeneity. Furthermore, traditional models rely on arbitrarily defined, prolonged preictal windows (e.g., 15–60 minutes), diluting the predictive learning signal. 

This repository provides the code and data to demonstrate how projecting electroencephalogram (EEG) signals into a **5-dimensional (5D) phase space**—coupled with **Feature-wise Linear Modulation (FiLM)**—can decouple individual baseline idiosyncrasies from the universal mechanics of ictogenesis, isolating a deterministic phase transition exactly 3.0 minutes prior to seizure onset.

## Repository Structure

The repository is organized into three main components: **Core Data**, **Core Algorithms (Data Extraction & Modeling)**, and **Plotting Scripts**.

### 1. Core Data (CSV files)
These files contain the processed 5D phase space features extracted from the CHB-MIT Scalp EEG database and the subsequent ablation results.
* `Data_Fig1_Raw_Delta_Tau.csv` : Raw tracking data containing Minimum Sample Entropy, $\tau_{TE}$, $\tau_{SVM}$, and $\Delta\tau$ across Normal, Inactive, and Active Focal epochs (150 epochs total). Used to construct the cohort dynamics and topological attractors in Fig. 1.
* `Data_Fig2a_Static_2D.csv` : Prediction accuracy (ROC-AUC) across 9 unseen patients using an uncorrected static 2D feature model.
* `Data_Fig2b_Baseline_Tracking.csv` : Continuous tracking data demonstrating the spatial fragmentation of baseline states across multiple individuals.
* `Data_Fig2c_Unconditioned_5D.csv` : Prediction accuracy using the expanded 5D feature set *without* dynamic baseline conditioning.
* `Data_Fig2d_Overfitting.csv` : Trajectory of Train vs. Test ROC-AUC over 20 epochs for an unconditioned model, demonstrating the catastrophic generalization gap.
* `Data_Fig2e_Conditioned_5D_7.5min.csv` : Prediction accuracy using the fully conditioned model evaluated at an arbitrarily prolonged preictal window (7.5 minutes).
* `Data_Fig3_Time_Ablation.csv` : Time-to-seizure ablation results, tracking mean predictive accuracy from 15.0 to 1.0 minutes prior to clinical onset.

### 2. Core Algorithms (Python files)
These scripts constitute the main computational pipeline described in the Methods section of the manuscript.
* `extract_5d_features.py` : MNE-Python based preprocessing pipeline. It extracts the 5D phase space features (Minimum Sample Entropy, Transfer Entropy via JIT-compiled KSG estimators, and spatial classification time via Riemannian geometry on covariance matrices) from raw `.edf` files.
* `subject_conditioned_conformer.py` : PyTorch implementation of the deep learning architecture. It integrates the FiLM (Feature-wise Linear Modulation) layer to dynamically scale and shift the Conformer blocks based on the patient-specific 5D baseline vector. Includes the training and evaluation loop with AdamW optimization.

### 3. Reproducibility & Plotting Scripts (Python files)
These scripts independently load the finalized CSV data and generate the publication-ready figures (PDF format, adhering to *Nature Communications* style guidelines).
* `plot_fig1_baseline_dynamics.py` : Performs K-Means clustering, Welch's t-tests with Benjamini-Hochberg FDR correction, and generates the spatial classification time ($\tau_{SVM}$) boxplots (Fig. 1a).
* `plot_fig2_generalization.py` : Generates the statistical plots illustrating the generalization gap, spatial fragmentation, and ablation results (Fig. 2a, b, c, d, e).
* `plot_fig3_time_ablation.py` : Generates the time-to-seizure ablation trajectory identifying the 3.0-minute deterministic phase transition (Fig. 3).

## Requirements & Environment Setup

To ensure exact reproducibility, it is recommended to execute the code within a dedicated Python environment. The primary dependencies are:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn torch mne numba statsmodels



## How to Reproduce the Main Findings

**Step 1: Verify the Data**
Ensure all 7 CSV files listed above are located in the root directory.

**Step 2: Reproduce the Statistical Analysis and Figures**
Execute the plotting scripts directly. They will parse the CSV data, run the defined statistical tests, and output high-resolution `.pdf` files.

```bash
python plot_fig1_baseline_dynamics.py
python plot_fig2_generalization.py
python plot_fig3_time_ablation.py

```

**Step 3: (Optional) Re-run the Core Processing Pipeline**
If you wish to re-extract the 5D features from scratch:

1. Download the raw CHB-MIT Scalp EEG dataset from PhysioNet (https://physionet.org/content/chbmit/).
2. Place the dataset in a directory named `CHBMIT_Data` within the repository root.
3. Run `python extract_5d_features.py` to regenerate the tracking data.
4. Run `python subject_conditioned_conformer.py` to retrain the model and evaluate cross-patient generalization.

## Code Availability Policy

This repository has been established to fully comply with the *Nature Portfolio* policies on code and data availability, ensuring maximum transparency, reproducibility, and utility for the broader computational neuroscience community.
