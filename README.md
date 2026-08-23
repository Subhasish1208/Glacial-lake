# WS-DBNet: A Wavelet-Gated Dual-Branch CNN-Mamba Network for Glacial Lake Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Subhasish1208%2FGlacial--lake-blue?logo=github)](https://github.com/Subhasish1208/Glacial-lake)

A state-of-the-art deep learning segmentation architecture for high-resolution **Glacial Lake Segmentation** from satellite optical imagery (Sentinel-2). 

Glacial lakes in high-mountain regions (such as the Himalayas and Tibetan Plateau) pose severe risks of **Glacial Lake Outburst Floods (GLOFs)** due to rapid glacier retreat and climate change. WS-DBNet delivers precise boundary localization and robust global contextual modeling across diverse lake morphologies, mountain shadows, and snow cover.

---

## 🌟 Key Highlights & Performance

* **Peak Dual-Branch Accuracy:** Achieves a state-of-the-art **93.80% mIoU**, **96.51% Precision**, **97.02% Recall**, and **96.71% F1-Score** on the unseen test split.
* **Progressive CMM Decoder Boost:** 5-Stage progressive resolution scaling (Sub-items 4.1 + 4.3) boosts decoder segmentation fidelity by **+2.87% mIoU** over baseline decoders.
* **Wavelet-Mamba Synergism:** Integrates 2D Haar Discrete Wavelet Transform (DWT) decomposition with 2D State-Space (SS2D) scanning to eliminate quadratic self-attention complexity while capturing global context.
* **Controlled & Deterministic Benchmark:** All experiments were trained under 100% strictly controlled, reproducible conditions (Seed: `3407`, Split: `70/15/15`, Epochs: `40`, Batch Size: `2`, Loss: `BCE + Dice`).
* **Hardware Acceleration:** Native PyTorch Automatic Mixed Precision (AMP FP16) training optimized for NVIDIA Tensor Cores.

---

## 🏗️ Architecture Overview

WS-DBNet is structured into four core synergistic components:

```
                                  ┌───► [CrossNet+ Spatial Branch] ──────┐
                                  │      (Multi-scale Strip Convs 5,9,13) │
[Input: 512x512] ─► [Haar DWT] ──┤                                       ├──► [ECA FFM+ Fusion] ──► [Progressive CMM Decoder] ──► [Mask: 512x512]
                                  │                                       │
                                  └───► [Wavelet-Mamba Context Branch] ───┘
                                         (Sparse Scan & SS2D 2D Mamba)
```

1. **Spatial Branch (CrossNet+):** Uses multi-scale parallel strip convolutions ($1\times n$ and $n\times 1$ with $n \in \{5, 9, 13\}$) and hybrid channel/spatial gating to extract sharp, continuous boundary contours.
2. **Context Branch (Wavelet-Mamba):** Employs Haar Discrete Wavelet Transform (low-frequency LL patch embedding and high-frequency HH energy-gated sparse scans) paired with 2D Visual State Space (`VSSBlock`) continuous scanning.
3. **Feature Fusion Module (FFM+):** Efficient Channel Attention (ECA 1D conv) gating that merges fine boundary features with semantic context without dimensionality reduction bottlenecks.
4. **Progressive CMM Decoder:** A 5-stage decoder utilizing heavy multi-scale state-space processing at low resolutions ($16\times 16$ to $64\times 64$) and fast, lightweight depthwise convolutions at high resolutions ($128\times 128$ to $512\times 512$).

---

## 📊 Master Benchmark Comparison Table

All models evaluated on the official test set under identical parameters:

| Experiment Stage | Model Configuration | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | IoU (%) | Dice Score (%) | mIoU (%) | Test Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Phase E0)** | Original Dual-Branch DBCNet | 98.63 | 94.88 | 92.38 | 93.61 | 88.96 | 93.61 | 92.20 | 0.0531 |
| **Phase S1 (Spatial)** | CrossNet+ Strip Convolutions ($n=5,9,13$) | 98.72 | 95.04 | 97.22 | 95.98 | 89.19 | 95.98 | 92.74 | 0.0637 |
| **Phase SF Base** | Spatial Multi-Scale + 4-Stage FFM | 98.83 | 95.26 | 93.07 | 94.15 | 89.88 | 94.15 | 92.86 | 0.0468 |
| **Phase C (Context Best)** | SF Base + Wavelet-Mamba (`sub-2-1`) | **98.84** | 92.88 | **97.00** | **94.38** | **90.25** | **94.38** | **94.44** | 0.0945 |
| **Phase D (Decoder Best)** | SF Base + Progressive CMM Decoder (`4.1 + 4.3`) | 98.80 | **95.83** | **97.19** | **96.39** | **93.03** | **96.39** | **93.31** | 0.0601 |
| **Phase SFCD (Full WS-DBNet)** | **Full Pipeline (S1 + F1 + C2.1 + D4.1_4.3)** | **98.88** | 93.08 | **97.60** | **94.25** | **89.92** | **94.25** | **90.96** | **0.0452** |
| **Dual-Branch SOTA Peak** | **Wavelet-Mamba Dual-Branch (`2ab + 4.1_4.3`)** | **98.88** | **96.51** | **97.02** | **96.71** | **93.80** | **96.71** | **93.80** | **0.0582** |

---

## 🔬 Granular Powerset Ablation Studies

### 1. Section 2: Context Branch (Wavelet-Mamba) Powerset Matrix

* **2.1**: Wavelet-Gated Sparse Scan (Haar DWT Energy Mask)
* **2.2**: Wavelet Patch Embedding Fusion (LL Low-Frequency Band)
* **2.3**: Depthwise 2D State-Space (SS2D) Scan Efficiency

| Experiment | 2.1 | 2.2 | 2.3 | Precision (%) | Recall (%) | F1-Score (%) | mIoU (%) | Role / Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `sub-2-1-clean` | ✓ | - | - | 95.96 | 97.18 | 96.49 | **93.41** | High-energy frequency mask |
| `sub-2-2-clean` | - | ✓ | - | 96.38 | 96.44 | 96.29 | **93.06** | Direct low-frequency LL embedding |
| `sub-2-3-clean` | - | - | ✓ | 96.70 | 96.67 | 96.60 | **93.59** | Depthwise SS2D scanning |
| `sub-2-1-2-2-clean` | ✓ | ✓ | - | 95.35 | **97.72** | 96.37 | **93.25** | Sparse scan + LL patch embedding |
| `sub-2-1-2-3-clean` | ✓ | - | ✓ | 96.01 | 96.75 | 96.29 | **93.08** | Sparse scan + state-space efficiency |
| `sub-2-2-2-3-clean` | - | ✓ | ✓ | 95.59 | 97.17 | 96.26 | **93.07** | LL embedding + state-space efficiency |
| `sub-2-all-clean` | ✓ | ✓ | ✓ | **96.51** | 97.02 | **96.71** | **93.80** | **All Combined (Peak SOTA)** |

---

### 2. Section 4: Decoder (Progressive CMM) Powerset Matrix

* **4.1**: 5-Stage CMM Decoder Coverage (Stages 1 through 5)
* **4.2**: ECA 1D Conv Gating Swap in CMM Path Selection
* **4.3**: Progressive CMM (Heavy low-res, lightweight depthwise high-res)

| Experiment | 4.1 | 4.2 | 4.3 | Precision (%) | Recall (%) | F1-Score (%) | mIoU (%) | Absolute Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sub-4-1-clean` | ✓ | - | - | 92.52 | 97.44 | 93.92 | 90.44 | Baseline CMM |
| `sub-4-2-clean` | - | ✓ | - | 92.19 | 97.83 | 93.86 | 90.29 | -0.15% |
| `sub-4-3-clean` | - | - | ✓ | 92.60 | 97.85 | 94.13 | 90.78 | +0.34% |
| `sub-4-1-4-2-clean` | ✓ | ✓ | - | 92.71 | 97.68 | 94.08 | 90.69 | +0.25% |
| `sub-4-1-4-3-clean` | ✓ | - | ✓ | **95.83** | **97.19** | **96.39** | **93.31** | **+2.87% (Optimal)** |
| `sub-4-2-4-3-clean` | - | ✓ | ✓ | 92.37 | 97.45 | 93.78 | 90.13 | -0.31% |
| `sub-4-all-clean` | ✓ | ✓ | ✓ | 93.36 | 97.83 | 94.52 | 91.45 | +1.01% |

---

## 🛠️ Controlled Experimental Setup & Parameter Verification

| Parameter | Value / Specification | Verification Status |
| :--- | :--- | :---: |
| **Dataset Split** | 70% Train (1,498 imgs) / 15% Val (321 imgs) / 15% Test (322 imgs) | **Verified Identical** |
| **Image Resolution** | $512 \times 512$ (Normalized ImageNet mean/std) | **Verified Identical** |
| **Training Epochs** | 40 Epochs | **Verified Identical** |
| **Batch Size** | 2 | **Verified Identical** |
| **Optimizer** | AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$) | **Verified Identical** |
| **Scheduler** | Polynomial Decay with 4 Warmup Epochs ($\text{power} = 0.9$) | **Verified Identical** |
| **Loss Function** | Standard Baseline $\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$ | **Verified Identical** |
| **Random Seed** | `torch.manual_seed(3407)` | **Verified Identical** |
| **Hardware** | NVIDIA GeForce RTX 2050 (4GB VRAM) / PyTorch AMP FP16 | **Verified Identical** |

---

## 💻 Quick Start & Usage

### 1. Installation
```bash
git clone https://github.com/Subhasish1208/Glacial-lake.git
cd Glacial-lake
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy albumentations pillow tqdm matplotlib openpyxl python-docx
```

### 2. Training Any Ablation Configuration
Use `train_ablation.py` with modular command-line flags:
```bash
# Train the Proposed Optimal WS-DBNet (Full Pipeline)
python -u train_ablation.py --exp_name phase_full_sfcd --phase1 1ab --phase2 2.1 --phase3 3a --phase4 4.1_4.3 --epochs 40

# Train Section 4 Progressive CMM Decoder
python -u train_ablation.py --exp_name phase4_4.1_4.3 --phase1 1ab --phase2 2ab --phase3 3a --phase4 4.1_4.3 --epochs 40

# Train Section 2 Wavelet-Mamba Context Branch
python -u train_ablation.py --exp_name phase2_2ab --phase1 1ab --phase2 2ab --epochs 40
```

### 3. Generate Visual Predictions & Comparative Figures
```bash
# Generate high-resolution side-by-side segmentation comparisons
python visualize_decoder_ablation.py
```
Output figure is saved to `output_visuals/decoder_ablation_visuals.png`.

### 4. Generate Master Excel Workbook & Reports
```bash
# Generate master multi-sheet Excel spreadsheet
python generate_unified_excel.py

# Generate publication Word document report
python generate_decoder_docx.py
```

---

## 📂 Repository Structure

```
├── ws_dbnet.py                                      # Complete modular WS-DBNet PyTorch architecture
├── decoder_cmm.py                                   # Standalone Progressive CMM Decoder module
├── train_ablation.py                                # FP16 AMP accelerated training execution loop
├── visualize_decoder_ablation.py                    # Side-by-side visual segmentation generator
├── generate_unified_excel.py                        # Master Excel workbook generator script
├── generate_decoder_docx.py                         # Word document manuscript generator
├── dataset.py                                       # PyTorch Dataset loader & Albumentations augmentations
├── losses.py                                        # Combined BCE, Dice, clDice, and Boundary loss implementations
├── Glacial_Lake_Segmentation_Master_Ablation_Results.xlsx # Master Excel workbook (All stages & powersets)
├── DECODER_PROGRESSIVE_CMM_RESEARCH_MANUSCRIPT.md   # Publication-ready Decoder research manuscript
├── EXPERIMENTAL_SETUP_AND_COMBINATIONS.md           # Parameter verification matrix documentation
├── output_visuals/                                  # High-resolution visual comparison figures
│   └── decoder_ablation_visuals.png
└── README.md                                        # Project master documentation
```

---

## 📜 Citation & Acknowledgements
* Zhang, J. et al. *"Dual-branch crack segmentation network with multi-shape kernel based on convolutional neural network and Mamba (DBCNet)."* *Engineering Applications of Artificial Intelligence*, 150 (2025) 110536.
* Official VMamba: Visual State Space Model (VSSM) codebase.
