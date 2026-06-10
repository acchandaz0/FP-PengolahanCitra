╔══════════════════════════════════════════════════════════════════════════════╗
║                    MMSK 3D U-Net Project Structure                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. PREPROCESSING (di luar folder mmsk)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
   
   Input: braTS/training_data1_v2/BraTS-GLI-xxxxx-xxx/
          ├── *-t1n.nii.gz  (T1)
          ├── *-t2w.nii.gz  (T2)
          ├── *-t2f.nii.gz  (FLAIR)
          ├── *-t1c.nii.gz  (T1ce)
          └── *-seg.nii.gz  (Segmentation)
   
   Process: preprocessing_template.py
            ├── Load 4 modalities
            ├── Z-score normalization
            ├── Crop non-zero region
            └── Pad to 128×128×128
   
   Output: braTS/preprocessed/BraTS-GLI-xxxxx-xxx.npz
           ├── images: [4, D, H, W]
           └── seg: [D, H, W]

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. DATASET PREPARATION (mmsk/prepare_dataset.py)                            │
└─────────────────────────────────────────────────────────────────────────────┘
   
   Input: braTS/preprocessed/*.npz
   
   Process: prepare_dataset.py
            ├── Scan all .npz files
            ├── Split train/val/test (70/10/20)
            ├── Stratified by tumor grade
            └── Generate metadata
   
   Output: mmsk/dataset.json
           {
             "train": [...],
             "val": [...],
             "test": [...]
           }

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. MODEL ARCHITECTURES                                                      │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │ A. mmsk_3d_unet.py ⭐ (PROPOSED MODEL)                               │
   └──────────────────────────────────────────────────────────────────────┘
   
   Input: [B, 4, D, H, W]
          ↓
   ┌──────────────────────┐
   │  Encoder (4 levels)  │  ← MMSK Blocks
   │  ├─ MMSK Block 1     │     ├─ 2 branches (d=1, d=2)
   │  ├─ MMSK Block 2     │     ├─ Cross-modal gating
   │  ├─ MMSK Block 3     │     └─ Adaptive selection
   │  └─ MMSK Block 4     │
   └──────────────────────┘
          ↓
   ┌──────────────────────┐
   │  Bottleneck          │  ← MMSK Block
   └──────────────────────┘
          ↓
   ┌──────────────────────┐
   │  Decoder (4 levels)  │  ← MMSK Blocks + Skip Connections
   │  ├─ MMSK Block 1     │
   │  ├─ MMSK Block 2     │
   │  ├─ MMSK Block 3     │
   │  └─ MMSK Block 4     │
   └──────────────────────┘
          ↓
   Output: [B, 4, D, H, W] (Background, WT, TC, ET)
   
   
   ┌──────────────────────────────────────────────────────────────────────┐
   │ B. baseline_3d_unet_brats.py (BASELINE 1)                            │
   └──────────────────────────────────────────────────────────────────────┘
   
   Standard 3D U-Net
   ├─ NO Selective Kernel
   ├─ NO Cross-modal gating
   └─ Standard Conv3D blocks
   
   
   ┌──────────────────────────────────────────────────────────────────────┐
   │ C. sk_3d_unet_tversky_brats.py (BASELINE 2)                          │
   └──────────────────────────────────────────────────────────────────────┘
   
   SK 3D U-Net (Li et al., CVPR 2019)
   ├─ ✅ Selective Kernel
   ├─ ❌ NO Cross-modal gating
   └─ Attention from feature map only
   
   
   ┌──────────────────────────────────────────────────────────────────────┐
   │ D. bgsk_3d_unet.py (ABLATION)                                        │
   └──────────────────────────────────────────────────────────────────────┘
   
   Background-aware SK
   ├─ ✅ Selective Kernel
   ├─ ✅ Background-specific attention
   └─ ❌ NO Full cross-modal gating

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. TRAINING SCRIPTS                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │ train_proposed_mmsk_tversky.py ⭐                                    │
   └──────────────────────────────────────────────────────────────────────┘
   
   Model: MMSK3DUNet
   Loss: Tversky (α=0.3, β=0.7)
   Optimizer: AdamW (lr=1e-4)
   Batch: 2
   Epochs: 100
   
   Output:
   ├─ robustmmsk/best_model.pth
   ├─ robustmmsk/checkpoint_epoch_*.pth
   └─ results/training_log.csv
   
   
   ┌──────────────────────────────────────────────────────────────────────┐
   │ train_baseline2_unet_tversky.py                                      │
   └──────────────────────────────────────────────────────────────────────┘
   
   Model: SK3DUNet (Baseline 2)
   Loss: Tversky
   
   
   ┌──────────────────────────────────────────────────────────────────────┐
   │ train_ablation2_mmsk_dice.py                                         │
   └──────────────────────────────────────────────────────────────────────┘
   
   Model: MMSK3DUNet
   Loss: Dice (untuk ablation study)

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. EVALUATION & VISUALIZATION                                               │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │ visualize_mmsk_attention.py                                          │
   └──────────────────────────────────────────────────────────────────────┘
   
   Input: robustmmsk/best_model.pth
   
   Visualizations:
   ├─ Attention weights (α₁, α₂)
   ├─ Cross-modal gate values
   ├─ Heatmaps overlay on MRI
   └─ Per-region analysis (ET, WT, TC)
   
   Output: results/attention_maps/

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. COMPARISON TABLE                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

   ╔═══════════════════╦════╦═════════════╦═════════╦═══════════════════════╗
   ║ Model             ║ SK ║ Cross-Modal ║ Loss    ║ File                  ║
   ╠═══════════════════╬════╬═════════════╬═════════╬═══════════════════════╣
   ║ MMSK (Proposed) ⭐║ ✅ ║ ✅          ║ Tversky ║ mmsk_3d_unet.py       ║
   ║ Baseline 1        ║ ❌ ║ ❌          ║ Tversky ║ baseline_3d_unet...   ║
   ║ Baseline 2 (SK)   ║ ✅ ║ ❌          ║ Tversky ║ sk_3d_unet_tversky... ║
   ║ BGSK (Ablation)   ║ ✅ ║ Partial     ║ Tversky ║ bgsk_3d_unet.py       ║
   ║ Ablation (Dice)   ║ ✅ ║ ✅          ║ Dice    ║ (same model)          ║
   ╚═══════════════════╩════╩═════════════╩═════════╩═══════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. COMPLETE WORKFLOW                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

   Step 1: Preprocessing
   ─────────────────────
   $ python3 preprocessing_template.py
   
   Step 2: Prepare Dataset
   ────────────────────────
   $ cd mmsk
   $ python3 prepare_dataset.py
   
   Step 3: Train Models
   ─────────────────────
   # Proposed MMSK
   $ python3 train_proposed_mmsk_tversky.py
   
   # Baseline 1
   $ python3 train_baseline2_unet_tversky.py
   
   # Ablation
   $ python3 train_ablation2_mmsk_dice.py
   
   Step 4: Visualize
   ──────────────────
   $ python3 visualize_mmsk_attention.py --checkpoint robustmmsk/best_model.pth
   
   Step 5: Compare Results
   ────────────────────────
   $ ls results/
     ├── mmsk_tversky/
     ├── baseline_tversky/
     └── mmsk_dice/

╔══════════════════════════════════════════════════════════════════════════════╗
║                              KEY INNOVATIONS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

   1. Cross-Modal Gating
      ├─ T1ce → ET (Enhancing Tumor)
      ├─ FLAIR → WT (Whole Tumor)
      └─ T1+T1ce → TC (Tumor Core)
   
   2. Adaptive Receptive Field
      ├─ Small kernel (d=1) → Fine details
      └─ Large kernel (d=2) → Diffuse boundaries
   
   3. Tversky Loss
      ├─ α=0.3, β=0.7
      └─ Better for class imbalance
