# Glacial Lake Segmentation using DBCNet (vs. DeepLabV3)

A deep learning project comparing a state-of-the-art **DBCNet** (Dual-Branch CNN-Mamba Network) architecture against a standard **DeepLabV3** baseline for segmenting glacial lakes from Sentinel-2 optical imagery.

Monitoring glacial lakes is crucial for early warning systems against Glacial Lake Outburst Floods (GLOFs) and analyzing climate change impacts in high-mountain regions (like the Himalayas).

---

## 🌟 Key Highlights
* **DBCNet Performance:** Achieved a **93.03% mIoU** and **96.25% F1-score** on unseen test data.
* **DeepLabV3 Comparison:** Outperformed standard DeepLabV3 by **+5.43% mIoU** and **+4.01% F1-score**.
* **GPU Acceleration:** Fully optimized and trained using PyTorch with CUDA 12.6 support on an NVIDIA RTX 2050.

---

## 🛠️ The Architecture (DBCNet)

DBCNet is a "dual-branch" network, meaning it processes the input satellite image in two different ways at the same time to get the best of both worlds:

```
                  ┌───► [Spatial Branch: CrossNet] ────┐
                  │      (Captures fine edges/shapes)  │
[Input: 512x512] ─┤                                    ├──► [Feature Fusion (FFM)] ──► [Decoder (CMM)] ──► [Mask: 512x512]
                  │                                    │
                  └───► [Context Branch: VMamba] ──────┘
                         (Captures global structure)
```

### 1. Spatial Branch (CrossNet)
* **Goal:** Focuses on textures, edges, and small-scale details.
* **How it works:** Uses `CrossBlock`s. Unlike regular 3x3 convolutions, a `CrossBlock` runs three convolutions in parallel: a standard 3x3, a horizontal 1x9, and a vertical 9x1. Combining these allows the model to capture features in multiple directions and shapes without losing boundary accuracy.

### 2. Context Branch (VMamba / SS2D)
* **Goal:** Focuses on the "big picture" (the global layout and landscape context).
* **How it works:** Utilizes Visual State Space blocks (`VSSBlock`). It approximates the 2D Selective Scan (SS2D) mechanism from Mamba. This scans the image in multiple directions to establish connections between distant parts of the image with linear time complexity.
* *Note: To make installation hassle-free and avoid Windows compilation errors, we implemented a custom pure-PyTorch `SS2D_Approximation`.*

### 3. Feature Fusion Module (FFM)
* Combines the features from the Spatial (CNN) branch and Context (Mamba) branch.
* Uses **Squeeze-and-Excitation (SE) Attention** to automatically decide which features are more important and scale them accordingly.

### 4. Decoder (CMM)
* Uses **Cross-aware Mamba Module (CMM)** blocks to scale the low-resolution features back up to the original 512x512 image size while retaining edge details.

---

## 📏 Feature Map Dimensions (Shapes)
Here is how the shape of a single batch of images (`[Batch, Channels, Height, Width]`) flows through DBCNet:

1. **Input:** `[B, 3, 512, 512]` (RGB imagery)
2. **Layer 1:** `[B, 16, 512, 512]`
3. **Layer 2 (Downsample):** `[B, 32, 256, 256]`
4. **Layer 3 (Downsample & FFM):** `[B, 64, 128, 128]`
5. **Layer 4 (Downsample & FFM):** `[B, 128, 64, 64]`
6. **Layer 5 (Downsample & FFM):** `[B, 256, 32, 32]`
7. **Layer 6 (Mamba Context only):** `[B, 512, 16, 16]`
8. **Decoder (Upsampling steps back to input size):**
   * Up-level 5: `[B, 256, 32, 32]`
   * Up-level 4: `[B, 128, 64, 64]`
   * Up-level 3: `[B, 64, 128, 128]`
   * Up-level 2: `[B, 32, 256, 256]`
   * Up-level 1: `[B, 16, 512, 512]`
9. **Output:** `[B, 1, 512, 512]` (Binary segment mask where `1` = Glacial Lake, `0` = Background)

---

## 📊 Evaluation Results (Test Set)

We split the dataset into 70% Training, 15% Validation, and 15% Testing. Both models were trained for 80 epochs with identical hyperparameters (AdamW, PolyLR, Warmup) on the GPU.

| Model | Precision | Recall | F1-Score | mIoU |
| :--- | :---: | :---: | :---: | :---: |
| **DBCNet (Ours)** | **95.34%** | **97.43%** | **96.25%** | **93.03%** |
| **DeepLabV3 (Baseline)** | 91.39% | 95.11% | 92.24% | 87.60% |
| **Difference** | **+3.95%** | **+2.32%** | **+4.01%** | **+5.43%** |

### Visual Comparisons
The comparison results saved in `output_visuals/comparison_results.png` show:
* **DBCNet (Red Overlay)** cleanly matches the boundaries and successfully isolates smaller lakes.
* **DeepLabV3 (Blue Overlay)** occasionally creates fragmented components or over-segments muddy/shadowy terrains.

---

## 💻 How to Run the Code

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy albumentations pillow tqdm matplotlib
```

### 2. Dataset
Prepare your dataset. The images and ground truth masks should be located in a dataset directory containing `images` and `masks` folders. Update the `data_dir` path in the scripts to point to your directory.

### 3. Data Audit & Splits
Run the audit script to check dimensions, compute global mean/std for normalizations, and generate the dataset splits:
```bash
python data_audit.py
```

### 4. Training
To train the architectures on your GPU:
```bash
# To train DBCNet
python train.py

# To train DeepLabV3
python train_deeplabv3.py
```

### 5. Evaluation & Visualization
To generate test set statistics and comparative overlays:
```bash
# Evaluate test metrics
python evaluate.py
python evaluate_deeplabv3.py

# Generate comparison image grids
python visualize_compare.py
```

---

## 📂 Repository Structure
```
├── dataset.py               # Custom PyTorch dataset & data augmentations
├── dbcnet.py                # Full DBCNet model architecture
├── deeplabv3_model.py       # DeepLabV3 baseline wrapper
├── train.py                 # DBCNet training loop
├── train_deeplabv3.py       # DeepLabV3 training loop
├── evaluate.py              # DBCNet test set evaluation
├── evaluate_deeplabv3.py    # DeepLabV3 test set evaluation
├── visualize_compare.py     # Comparison visualization script
├── data_audit.py            # Phase 0 dataset auditing & statistics
└── README.md                # Project documentation
```

---

## 📜 References
* Zhang, J. et al. "Dual-branch crack segmentation network with multi-shape kernel based on convolutional neural network and Mamba (DBCNet)." *Engineering Applications of Artificial Intelligence*, 150 (2025) 110536.
