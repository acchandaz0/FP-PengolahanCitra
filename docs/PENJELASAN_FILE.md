# Penjelasan File di Folder /home/mci/arsyadl/mmsk

## 📁 Struktur Folder

```
mmsk/
├── Model Architectures (Arsitektur Model)
│   ├── mmsk_3d_unet.py                    # ⭐ MODEL UTAMA (Proposed)
│   ├── baseline_3d_unet_brats.py          # Baseline 1: Standard 3D U-Net
│   ├── sk_3d_unet_tversky_brats.py        # Baseline 2: SK 3D U-Net
│   └── bgsk_3d_unet.py                    # Ablation: BGSK (Background-aware SK)
│
├── Training Scripts (Script Training)
│   ├── train_proposed_mmsk_tversky.py     # ⭐ TRAINING UTAMA
│   ├── train_baseline2_unet_tversky.py    # Training Baseline 2
│   ├── train_bgsk_3d_unet_brats.py        # Training BGSK
│   └── train_ablation2_mmsk_dice.py       # Ablation study (Dice loss)
│
├── Utilities (Utilitas)
│   ├── prepare_dataset.py                 # Prepare dataset untuk training
│   ├── visualize_mmsk_attention.py        # Visualisasi attention maps
│   ├── dataset.json                       # Metadata dataset
│   └── requirements.txt                   # Dependencies
│
└── Output
    ├── results/                           # Hasil training & evaluasi
    └── robustmmsk/                        # Model checkpoints
```

---

## 📄 Penjelasan Detail Setiap File

### 1️⃣ **mmsk_3d_unet.py** ⭐ (MODEL UTAMA)

**Fungsi:** Implementasi arsitektur **Multi-Modal Selective Kernel 3D U-Net**

**Komponen Utama:**
```python
class MMSKConv3D:
    # Multi-Modal Selective Kernel Convolution
    # - 2 branch: kernel 3×3×3 (d=1) dan 3×3×3 (d=2)
    # - Cross-modal gating dari 4 modalitas MRI
    # - Adaptive receptive field selection
    
class MMSKBlock:
    # 2× MMSKConv3D berturut-turut
    
class MMSK3DUNet:
    # Encoder-Decoder dengan MMSK blocks
    # Input: [B, 4, D, H, W] (T1, T2, FLAIR, T1ce)
    # Output: [B, 4, D, H, W] (Background, WT, TC, ET)
```

**Inovasi:**
- ✅ Cross-modal gating: Modalitas MRI mempengaruhi pemilihan kernel
- ✅ Adaptive receptive field: Kernel kecil untuk detail, besar untuk boundary
- ✅ Sub-region aware: ET→T1ce, WT→FLAIR, TC→T1+T1ce

---

### 2️⃣ **train_proposed_mmsk_tversky.py** ⭐ (TRAINING UTAMA)

**Fungsi:** Script training untuk model MMSK dengan **Tversky Loss**

**Fitur:**
```python
# Loss Function
- Tversky Loss (α=0.3, β=0.7) untuk class imbalance
- Per-class weighting: WT, TC, ET

# Training Setup
- Optimizer: AdamW
- Learning rate: 1e-4 dengan ReduceLROnPlateau
- Batch size: 2 (karena 3D volume besar)
- Epochs: 100

# Data Augmentation
- Random flip, rotation
- Elastic deformation
- Intensity shift

# Evaluation Metrics
- Dice Score per class (WT, TC, ET)
- Hausdorff Distance 95%
- Sensitivity, Specificity
```

**Output:**
- Model checkpoints di `robustmmsk/`
- Training logs & metrics di `results/`

---

### 3️⃣ **baseline_3d_unet_brats.py** (BASELINE 1)

**Fungsi:** Standard 3D U-Net tanpa SK mechanism

**Arsitektur:**
```python
class Baseline3DUNet:
    # Standard U-Net architecture
    # - Encoder: 4 levels dengan max pooling
    # - Decoder: 4 levels dengan upsampling
    # - Skip connections
    # - NO selective kernel
    # - NO cross-modal gating
```

**Tujuan:** Baseline untuk membandingkan performa MMSK

---

### 4️⃣ **sk_3d_unet_tversky_brats.py** (BASELINE 2)

**Fungsi:** 3D U-Net dengan **Selective Kernel** (Li et al., CVPR 2019)

**Arsitektur:**
```python
class SKConv3D:
    # Standard SK (tanpa cross-modal gating)
    # - 2 branch: kernel 3×3×3 (d=1, d=2)
    # - Fuse: U = U₁ + U₂
    # - Select: Softmax attention dari GAP(U)
    # - NO modality-specific gating
```

**Perbedaan dengan MMSK:**
- ❌ Tidak ada cross-modal gating
- ❌ Tidak memanfaatkan informasi modalitas MRI
- ✅ Hanya attention dari feature map

---

### 5️⃣ **bgsk_3d_unet.py** (ABLATION STUDY)

**Fungsi:** Background-aware Selective Kernel

**Fitur:**
```python
class BGSKConv3D:
    # SK dengan background awareness
    # - Separate attention untuk background vs tumor
    # - Weighted fusion berdasarkan region
```

**Tujuan:** Ablation study untuk memahami kontribusi background modeling

---

### 6️⃣ **train_baseline2_unet_tversky.py**

**Fungsi:** Training script untuk SK 3D U-Net (Baseline 2)

**Setup:** Sama dengan training MMSK, tapi menggunakan model SK standar

---

### 7️⃣ **train_bgsk_3d_unet_brats.py**

**Fungsi:** Training script untuk BGSK model

---

### 8️⃣ **train_ablation2_mmsk_dice.py**

**Fungsi:** Ablation study - MMSK dengan **Dice Loss** (bukan Tversky)

**Tujuan:** Membandingkan efek loss function:
- Tversky Loss (proposed) vs Dice Loss
- Untuk membuktikan Tversky lebih baik handle class imbalance

---

### 9️⃣ **prepare_dataset.py**

**Fungsi:** Prepare dan split dataset untuk training

```python
# Fungsi:
1. Scan folder preprocessed data
2. Split train/val/test (70/10/20)
3. Stratified sampling by tumor grade
4. Generate dataset.json
```

**Output:** `dataset.json` dengan struktur:
```json
{
  "train": ["BraTS-GLI-00001-000.npz", ...],
  "val": [...],
  "test": [...]
}
```

---

### 🔟 **visualize_mmsk_attention.py**

**Fungsi:** Visualisasi attention maps dari MMSK

**Fitur:**
```python
# Visualisasi:
1. Attention weights per branch (α₁, α₂)
2. Cross-modal gate values
3. Overlay pada MRI slices
4. Comparison: ET vs WT vs TC regions

# Output:
- Heatmaps
- 3D volume rendering
- Attention distribution plots
```

**Tujuan:** Interpretability - memahami bagaimana model memilih kernel

---

### 1️⃣1️⃣ **dataset.json**

**Fungsi:** Metadata dataset hasil `prepare_dataset.py`

**Isi:**
```json
{
  "train": ["list of training samples"],
  "val": ["list of validation samples"],
  "test": ["list of test samples"],
  "num_classes": 4,
  "modalities": ["T1", "T2", "FLAIR", "T1ce"]
}
```

---

### 1️⃣2️⃣ **requirements.txt**

**Fungsi:** Dependencies Python

**Isi:**
```
torch>=1.10.0
nibabel
numpy
scipy
matplotlib
tqdm
```

---

## 🎯 Workflow Lengkap

### 1. **Preprocessing** (di luar folder mmsk)
```bash
python3 preprocessing_template.py
# Output: braTS/preprocessed/*.npz
```

### 2. **Prepare Dataset**
```bash
cd mmsk
python3 prepare_dataset.py
# Output: dataset.json
```

### 3. **Training**
```bash
# Model Utama (MMSK + Tversky)
python3 train_proposed_mmsk_tversky.py

# Baseline 1 (Standard U-Net)
python3 train_baseline2_unet_tversky.py

# Ablation (MMSK + Dice)
python3 train_ablation2_mmsk_dice.py
```

### 4. **Visualisasi**
```bash
python3 visualize_mmsk_attention.py --checkpoint robustmmsk/best_model.pth
```

---

## 📊 Perbandingan Model

| Model | SK | Cross-Modal | Loss | File |
|-------|----|-----------|----|------|
| **MMSK (Proposed)** | ✅ | ✅ | Tversky | `mmsk_3d_unet.py` |
| Baseline 1 | ❌ | ❌ | Tversky | `baseline_3d_unet_brats.py` |
| Baseline 2 (SK) | ✅ | ❌ | Tversky | `sk_3d_unet_tversky_brats.py` |
| BGSK | ✅ | Partial | Tversky | `bgsk_3d_unet.py` |
| Ablation | ✅ | ✅ | Dice | (sama, beda loss) |

---

## 🔑 File Paling Penting

1. **`mmsk_3d_unet.py`** - Arsitektur model utama
2. **`train_proposed_mmsk_tversky.py`** - Training script utama
3. **`prepare_dataset.py`** - Setup dataset
4. **`visualize_mmsk_attention.py`** - Interpretability

---

## 💡 Tips

- Mulai dari `prepare_dataset.py` untuk setup data
- Gunakan `train_proposed_mmsk_tversky.py` untuk training
- Bandingkan dengan baseline menggunakan `train_baseline2_unet_tversky.py`
- Visualisasi hasil dengan `visualize_mmsk_attention.py`
