# Glacial Lake Segmentation: DBCNet vs. VMambaSeg vs. DeepLabV3

A deep learning project comparing state-of-the-art **Mamba-based architectures** (**VMambaSeg** and **DBCNet**) against a standard **DeepLabV3** ResNet50 baseline for segmenting glacial lakes from Sentinel-2 optical imagery. 

Monitoring glacial lakes is crucial for early warning systems against Glacial Lake Outburst Floods (GLOFs) and analyzing climate change impacts in high-mountain regions (like the Himalayas).

---

## 🌟 Key Highlights
* **VMambaSeg Performance:** Achieved a state-of-the-art **93.65% mIoU** and **96.62% F1-score** on unseen test data.
* **DBCNet Performance:** Achieved a **93.03% mIoU** and **96.25% F1-score**.
* **Mamba vs. CNNs:** Both Mamba-based networks significantly outperform the standard DeepLabV3 CNN baseline (**+6.05%** and **+5.43% mIoU** respectively).
* **GPU Acceleration:** Fully optimized and trained using PyTorch with CUDA 12.6 support on an NVIDIA RTX 2050.
* **Windows Portability:** Implemented using a pure PyTorch State-Space scanning approximation, eliminating complex C++/CUDA kernel compilation issues (`mamba-ssm`) on Windows.

---

## 🛠️ Model Architectures

This project compares three models that process satellite imagery using different semantic levels:

### 1. VMambaSeg (Pure State-Space Encoder-Decoder)
* **Goal:** Employs a hierarchical State-Space backbone for feature extraction, paired with a standard skip-connection decoder.
* **Encoder:** A 4-stage hierarchical VMamba (VSSM) encoder. Input resolution is patch-embedded down to $128\times 128$. Each stage processes features using Visual State Space blocks (`VSSBlock`) which utilize a 2D selective scan approximation to capture long-range contextual relationships globally with linear complexity.
* **Decoder:** A U-Net style decoder that takes skip connections from stages 1, 2, and 3, upsamples the features via transpose convolutions, and fuses them to rebuild the spatial boundary details.

### 2. DBCNet (CNN-Mamba Hybrid)
* **Goal:** Merges spatial details (from a CNN branch) with global landscape context (from a Mamba branch).
* **Spatial Branch (CrossNet):** Uses `CrossBlock`s executing standard 3x3, horizontal 1x9, and vertical 9x1 convolutions in parallel to capture fine edges in multiple directions.
* **Context Branch (VMamba):** Captures the big picture using `VSSBlock`s to establish connections between distant parts of the image.
* **Fusion & Decoder:** Employs a Feature Fusion Module (FFM) with channel attention and a Cross-aware Mamba Module (CMM) decoder.

```
                  ┌───► [Spatial Branch: CrossNet] ────┐
                  │      (Captures fine edges/shapes)  │
[Input: 512x512] ─┤                                    ├──► [Feature Fusion (FFM)] ──► [Decoder (CMM)] ──► [Mask: 512x512]
                  │                                    │
                  └───► [Context Branch: VMamba] ──────┘
                         (Captures global structure)
```

### 3. DeepLabV3 ResNet50 (CNN Baseline)
* **Goal:** A standard fully-convolutional comparison baseline.
* **Details:** Uses ResNet50 for feature extraction and Atrous Spatial Pyramid Pooling (ASPP) to expand the receptive field using dilated convolutions.

---

## 📏 Feature Map Dimensions (Shapes)

### VMambaSeg Tensor Flow
1. **Input:** `[B, 3, 512, 512]` (Sentinel-2 RGB bands)
2. **Patch Embed:** `[B, 64, 128, 128]` (Stage 1 Skip Connection)
3. **Stage 2 (Downsample):** `[B, 128, 64, 64]` (Stage 2 Skip Connection)
4. **Stage 3 (Downsample):** `[B, 256, 32, 32]` (Stage 3 Skip Connection)
5. **Stage 4 (Downsample):** `[B, 512, 16, 16]` (Bottleneck)
6. **Decoder Steps:**
   * Up 3 (Concat with Stage 3 Skip): `[B, 512, 32, 32]` -> Conv -> `[B, 256, 32, 32]`
   * Up 2 (Concat with Stage 2 Skip): `[B, 256, 64, 64]` -> Conv -> `[B, 128, 64, 64]`
   * Up 1 (Concat with Stage 1 Skip): `[B, 128, 128, 128]` -> Conv -> `[B, 64, 128, 128]`
7. **Final Upsampling:**
   * Up to `[B, 32, 256, 256]` -> Conv -> `[B, 32, 256, 256]`
   * Up to `[B, 16, 512, 512]` -> Conv -> `[B, 16, 512, 512]`
8. **Output:** `[B, 1, 512, 512]` (Glacial Lake Segment Mask)

---

## 📊 Comparative Evaluation Results (Test Set)

We split the dataset into 70% Training (287 samples), 15% Validation (63 samples), and 15% Testing (63 samples) deterministically using saved split indices. All models were trained for 80 epochs with identical hyperparameters (AdamW, PolyLR, Warmup) on the RTX 2050 GPU.

| Model | Precision | Recall | F1-Score | mIoU | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VMambaSeg (Ours)** | **96.29%** | 97.12% | **96.62%** | **93.65%** | **Best Performance** |
| **DBCNet (CNN-Mamba)** | 95.34% | **97.43%** | 96.25% | 93.03% | Strong Runner-Up |
| **DeepLabV3 ResNet50** | 91.39% | 95.11% | 92.24% | 87.60% | Baseline |

### Key Conclusions:
* **VMambaSeg achieves the highest boundary accuracy (93.65% mIoU),** representing a **+0.62%** increase over DBCNet and a **+6.05%** improvement over the DeepLabV3 CNN baseline.
* **Higher Precision:** VMambaSeg suppresses false-positive noise from mountain shadows, clouds, and snow patches, reaching **96.29% Precision**.
* **Visual Verification:** Visual overlays show that VMambaSeg traces intricate lake edges and handles small isolated lakes with high structural integrity compared to DeepLabV3 which tends to fragment.

---

## 💻 How to Run the Code

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy albumentations pillow tqdm matplotlib
```

### 2. Dataset Setup
Ensure your Sentinel-2 dataset images and ground truth masks are located in a folder structure like:
```
glacial-lake-dataset/
├── images/     # 400x400 JPG/PNG images
└── masks/      # 400x400 ground truth labels (binary 0/255)
```
Update the `data_dir` variable in the training scripts to point to this directory.

### 3. Initialize Splits
Run the audit script to check data integrity, compute normalization parameters, and save the train/val/test splits:
```bash
python data_audit.py
```

### 4. Model Training
Run the training scripts for the respective models:
```bash
# Train VMambaSeg (Recommended)
python train_vmamba.py

# Train DBCNet
python train.py

# Train DeepLabV3 ResNet50
python train_deeplabv3.py
```

### 5. Model Evaluation & Visual Comparison
Calculate test split metrics and save comparative visualization grids:
```bash
# Evaluate model metrics
python evaluate_vmamba.py
python evaluate.py
python evaluate_deeplabv3.py

# Generate comparison plots (DBCNet vs. DeepLabV3 vs. VMambaSeg)
python visualize_compare_mamba.py
```

---

## 📂 Repository Structure
```
├── dataset.py                  # PyTorch custom dataset and augmentations
├── vmamba_seg.py               # VMambaSeg encoder-decoder architecture
├── dbcnet.py                   # DBCNet CNN-Mamba hybrid architecture
├── deeplabv3_model.py          # DeepLabV3 baseline model wrapper
├── train_vmamba.py             # VMambaSeg training pipeline
├── train.py                    # DBCNet training pipeline
├── train_deeplabv3.py          # DeepLabV3 training pipeline
├── evaluate_vmamba.py          # VMambaSeg test set evaluator
├── evaluate.py                 # DBCNet test set evaluator
├── evaluate_deeplabv3.py       # DeepLabV3 test set evaluator
├── visualize_compare_mamba.py  # Generates 3-way comparative plots
├── data_audit.py               # Pre-training dataset audit & splits generator
├── experiment_results.txt      # Text record of final set metrics
└── README.md                   # Project documentation
```

---

## 📜 References
* Zhang, J. et al. "Dual-branch crack segmentation network with multi-shape kernel based on convolutional neural network and Mamba (DBCNet)." *Engineering Applications of Artificial Intelligence*, 150 (2025) 110536.
* Official VMamba: Visual State Space Model (VSSM) codebase approximation.
