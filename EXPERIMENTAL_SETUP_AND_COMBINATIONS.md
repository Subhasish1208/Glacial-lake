# Controlled Experimental Setup & Structural Combination Matrix
### WS-DBNet: Glacial Lake Segmentation Study

This document verifies that **every single model combination** was trained under **100% controlled, identical training conditions** to ensure complete empirical fairness.

---

## 1. Experimental Parameter Verification Matrix

| Combination / Variant | Train / Val / Test Split | Epochs | Batch Size | Image Resolution | Optimizer | Learning Rate | Weight Decay | Scheduler | Primary Loss | Random Seed | Hardware / Precision | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S Only** (`phase1_1ab`) | 70 / 15 / 15 | 40 | 2 | $512 \times 512$ | AdamW | $0.001$ | $10^{-4}$ | Poly Warmup | BCE + Dice | 3407 | RTX 2050 (AMP FP16) | **Identical** |
| **C Only** (`phase2_2ab`) | 70 / 15 / 15 | 40 | 2 | $512 \times 512$ | AdamW | $0.001$ | $10^{-4}$ | Poly Warmup | BCE + Dice | 3407 | RTX 2050 (AMP FP16) | **Identical** |
| **S + C** (`phase2_2ab`) | 70 / 15 / 15 | 40 | 2 | $512 \times 512$ | AdamW | $0.001$ | $10^{-4}$ | Poly Warmup | BCE + Dice | 3407 | RTX 2050 (AMP FP16) | **Identical** |
| **S + FFM** (`phase3_3b`) | 70 / 15 / 15 | 40 | 2 | $512 \times 512$ | AdamW | $0.001$ | $10^{-4}$ | Poly Warmup | BCE + Dice | 3407 | RTX 2050 (AMP FP16) | **Identical** |
| **C + D** (`phase4_4_1`) | 70 / 15 / 15 | 40 | 2 | $512 \times 512$ | AdamW | $0.001$ | $10^{-4}$ | Poly Warmup | BCE + Dice | 3407 | RTX 2050 (AMP FP16) | **Identical** |
| **S + C + FFM** (`phase3_3a`) | 70 / 15 / 15 | 40 | 2 | $512 \times 512$ | AdamW | $0.001$ | $10^{-4}$ | Poly Warmup | BCE + Dice | 3407 | RTX 2050 (AMP FP16) | **Identical** |
| **S + C + D** (`phase4_4.1_4.3`) | 70 / 15 / 15 | 40 | 2 | $512 \times 512$ | AdamW | $0.001$ | $10^{-4}$ | Poly Warmup | BCE + Dice | 3407 | RTX 2050 (AMP FP16) | **Identical** |
| **S + C + FFM + D** (`phase4_all`) | 70 / 15 / 15 | 40 | 2 | $512 \times 512$ | AdamW | $0.001$ | $10^{-4}$ | Poly Warmup | BCE + Dice | 3407 | RTX 2050 (AMP FP16) | **Identical** |

---

## 2. Structural Combination Matrix (S, C, FFM, D) Results

| Combination | Spatial (S) | Context (C) | FFM (Fusion) | Decoder (D) | Precision (%) | Recall (%) | F1-Score (%) | mIoU (%) | Role / Experiment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **S Only** | $\checkmark$ | - | - | - | 95.04 | 97.22 | 95.98 | **92.74** | Multi-scale spatial boundary (`phase1_1ab`) |
| **C Only** | - | $\checkmark$ | - | - | 96.01 | 96.75 | 96.29 | **93.08** | Wavelet-Mamba context scan |
| **S + C** | $\checkmark$ | $\checkmark$ | - | - | **96.51** | 97.02 | **96.71** | **93.80** | Peak dual-branch encoder (`phase2_2ab`) |
| **S + FFM** | $\checkmark$ | - | $\checkmark$ | - | 94.82 | 97.33 | 95.79 | **92.36** | ECA fusion on spatial branch alone |
| **C + D** | - | $\checkmark$ | - | $\checkmark$ | 92.52 | 97.44 | 93.92 | **90.44** | Context branch to CMM decoder |
| **S + C + FFM** | $\checkmark$ | $\checkmark$ | $\checkmark$ | - | 95.70 | 97.72 | 96.60 | **93.61** | Dual-branch encoder + ECA fusion (`phase3_3a`) |
| **S + C + D** | $\checkmark$ | $\checkmark$ | - | $\checkmark$ | 94.58 | 96.88 | 95.40 | **91.81** | Dual-branch encoder to CMM decoder |
| **S + C + FFM + D** | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | 95.83 | **97.19** | 96.39 | **93.31** | **Full WS-DBNet Pipeline** (`phase4_4.1_4.3`) |
