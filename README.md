# DiffusionMaps_VS_PCA  
*Manifold Learning on Curved Data*

This project compares **Diffusion Maps** and **PCA** on nonlinear manifolds and real data. Using Swiss Roll, S-Curve, and the Breast Cancer Wisconsin dataset, it shows how geometry-aware embeddings outperform linear methods in both visualization and classification.

---

## What’s Inside

- Swiss Roll and S-Curve manifold generation  
- Full-kernel and kNN Diffusion Maps  
- PCA baseline  
- 2D embedding visualizations  
- kNN classification on embeddings  
- Confusion matrices, accuracy, and runtime comparisons  
- Breast cancer case study  
- Eigenvalue spectrum and diffusion coordinates (ψ)

---

## Why It Matters

PCA is linear and struggles with curved structure.  
Diffusion Maps preserve intrinsic geometry, producing cleaner embeddings and better downstream performance.

This repo demonstrates how nonlinear embeddings improve both insight and model accuracy.
