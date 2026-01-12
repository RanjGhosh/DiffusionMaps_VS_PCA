This project compares Diffusion Maps and Principal Component Analysis (PCA) as dimensionality reduction techniques on both synthetic manifolds and real-world data. Using Swiss Roll and S-Curve datasets, the code demonstrates how nonlinear structure can be recovered by Diffusion Maps—especially with a kNN kernel—while PCA struggles with curved manifolds.

The pipeline then extends to the Breast Cancer Wisconsin dataset, evaluating how PCA, full-kernel Diffusion Maps, and kNN Diffusion Maps affect downstream classification using a kNN classifier. The project visualizes embeddings, confusion matrices, eigenvalue spectra, runtime, and cross-validation accuracy.

Key Features

Synthetic manifold generation (Swiss Roll, S-Curve)

Full Gaussian and kNN-based Diffusion Maps

PCA baselines

2D embedding visualizations

kNN classification on embeddings

Confusion matrices, accuracy, and timing comparisons

Application to real biomedical data (Breast Cancer dataset)

Eigenvalue spectrum and diffusion coordinates (ψ) analysis

Goal

To show how geometry-aware, nonlinear embeddings outperform linear methods like PCA on curved data, and how this translates into improved classification performance on real datasets.

This repository serves as a compact, end-to-end study of manifold learning for both visualization and machine learning.
