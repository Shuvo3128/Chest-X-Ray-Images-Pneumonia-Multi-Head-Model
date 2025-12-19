Chest-X-Ray-Images-Pneumonia-Multi-Head-Model

Pneumonia Detection from Chest X-rays

Deep Learning with Explainable AI

Overview

This project implements a deep learning pipeline for binary classification of Pneumonia vs Normal chest X-ray images.
Multiple architectures are evaluated under identical settings, with a final focus on interpretability using Grad-CAM and multi-head supervision.

Dataset

Source: Kaggle Chest X-ray Pneumonia

Task: Binary classification (NORMAL vs PNEUMONIA)

Split: Stratified 70% / 15% / 15%

Input: 224×224, grayscale → RGB (ImageNet normalization)

Models

EfficientNet-B0

ResNet-50 (Single-head)

ResNet-50 (Multi-head – Final model)

Shared backbone

Main head + Auxiliary head

Loss:

𝐿
=
𝐿
𝑚
𝑎
𝑖
𝑛
+
0.4
×
𝐿
𝑎
𝑢
𝑥
L=L
main
	​

+0.4×L
aux
	​

⚙️ Training Setup

Optimizer: AdamW

Scheduler: ReduceLROnPlateau

Loss: Cross-Entropy

Regularization:

Strong data augmentation

MixUp + CutMix

Best model saved by Validation Accuracy

Results (Test Accuracy)
Model	Accuracy
EfficientNet-B0	~99%
ResNet-50	~98%
ResNet-50 Multi-head	~98%

The multi-head model preserves accuracy while providing improved interpretability.

Explainability

Grad-CAM implemented with automatic last-conv detection

Separate attention maps for main and auxiliary heads

Visualization includes:

Original image

MixUp / CutMix samples

Grad-CAM overlays

Difference heatmaps between heads
