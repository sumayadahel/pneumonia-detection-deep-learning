# pneumonia-detection-deep-learning
Official research repository for my M.Sc. thesis: Comparative Analysis of Interpretable Deep Learning Models for Pneumonia Detection in Chest X-rays
# Benchmarking Deep Learning Architectures, Attention Mechanisms, and XAI Frameworks for Pneumonia Detection in Chest X-Rays

[![M.Sc. Thesis Project](https://img.shields.io/badge/Thesis-M.Sc._Research-blue.svg)](https://github.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Official Research Codebase for Master's Thesis**  
> *Author:* [Your Name]  
> *Institution:* Ferhat Abbas University Setif 1, Faculty of Technology, Department of Electronics  

---

## 📌 Executive Summary

Pneumonia remains a leading cause of mortality worldwide, particularly among children and vulnerable populations. Automated Computer-Aided Diagnosis (CAD) systems powered by deep learning hold immense potential for rapid screening; however, clinical adoption is hindered by **class imbalance**, **domain shift**, and the **"black-box" nature of neural networks**.

This repository presents a systematic empirical evaluation of deep learning paradigms across two distinct radiological datasets:
1. **DS1 (Kaggle Pneumonia Dataset):** Binary classification (5,856 pediatric chest X-rays) evaluated for baseline and fine-grained attention architectures.
2. **DS2 (NIH ChestX-ray14 Subset):** Multi-class classification (Healthy, Pneumonia, Other Diseases; ~10,000 adult chest X-rays) challenging models with severe class imbalance and subtle clinical pathologies.

### Key Findings
* **Optimal Preprocessing:** Contrast Limited Adaptive Histogram Equalization (**CLAHE**) consistently outperformed raw images and complex filtering (e.g., Adaptive Masking), enhancing feature readability without losing structural boundaries.
* **Efficiency vs. Performance:** **Custom CNN with Squeeze-and-Excitation (SE)** attention achieved peak efficiency, yielding **97.50% Accuracy / 0.9798 F1-Score** on DS1 in just **336.56s** training time.
* **Generalization Leader:** **DenseNet121** attained the top raw performance across both DS1 (**98.08% Accuracy**) and DS2 (**78.36% Accuracy**).
* **Explainability Insight:** Combined **Grad-CAM++** and **KernelSHAP** visual evaluations proved that **CNN + SE** produces the most clinically plausible features—focusing sharply on lung parenchymal infiltrates rather than spurious background tokens.

---

## 🏗 System Architecture & Repository Layout

```text
pneumonia-detection-xai-benchmarking/
│
├── README.md                            # Executive research documentation
├── requirements.txt                      # Environment dependencies
├── LICENSE                              # Open-source MIT License
├── .gitignore                           # Ignored datasets, weights, and temp files
│
├── config.py                            # Hyperparameters, paths, and seed setup
├── train.py                             # Single CLI script for training all architectures
├── evaluate.py                          # Metric computation and plot generation
│
├── src/                                 # Core source code modules
│   ├── preprocessing/
│   │   ├── preprocessing_methods.py     # CLAHE, AHE, HE, High-Pass, Gaussian, Adaptive Masking
│   │   └── data_loader.py               # Dataset loaders, augmentation, & class rebalancing
│   │
│   ├── models/
│   │   ├── custom_cnn.py                # Baseline custom CNN architecture
│   │   ├── cnn_se.py                    # Custom CNN with Squeeze-and-Excitation
│   │   ├── attention_modules.py         # Self-Attention, CBAM, ECA, SAM, Attention Gate (AG)
│   │   ├── transfer_learning.py         # DenseNet121, ResNet50, EfficientNetB0
│   │   ├── transformers.py              # ViT-B/16, DeiT-B, Swin-B setups
│   │   └── hybrid_models.py             # CNN + ViT hybrid architecture
│   │
│   └── explainability/
│       ├── gradcam_engine.py            # Grad-CAM++ visualization engine
│       └── shap_engine.py               # SHAP feature importance mapping
│
├── notebooks/
│   └── Experimental_Logs.ipynb          # Interactive evaluation & visualization workspace
│
└── results/
    ├── metrics/                         # Exported metric tables (CSV format)
    └── figures/                         # High-resolution thesis figures & XAI overlays
