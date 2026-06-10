# Panduan Preprocessing untuk MMSK (Tanpa Split)

## 🎯 Struktur Data BraTS

BraTS sudah menyediakan split terpisah:
```
braTS/
├── training_data1_v2/      # Data training
│   ├── BraTS-GLI-00001-000/
│   ├── BraTS-GLI-00001-001/
│   └── ...
└── validation_data/        # Data validation
    ├── BraTS-GLI-xxxxx-xxx/
    └── ...
```

## 📝 Workflow yang Benar

### 1. Preprocess Training Data
```python
# Di preprocessing_mmsk.ipynb, ubah:
DATA_DIR = Path("/home/mci/arsyadl/braTS/training_data1_v2")
OUTPUT_DIR = Path("/home/mci/arsyadl/braTS/preprocessed_train")
```

### 2. Preprocess Validation Data
```python
# Di preprocessing_mmsk.ipynb, ubah:
DATA_DIR = Path("/home/mci/arsyadl/braTS/validation_data")
OUTPUT_DIR = Path("/home/mci/arsyadl/braTS/preprocessed_val")
```

### 3. Generate dataset.json
```bash
cd /home/mci/arsyadl/mmsk
python3 prepare_dataset_no_split.py
```

## 📁 Output Structure

```
braTS/
├── preprocessed_train/     # Hasil preprocess training
│   ├── BraTS-GLI-00001-000.npz
│   ├── BraTS-GLI-00001-001.npz
│   └── ...
└── preprocessed_val/       # Hasil preprocess validation
    ├── BraTS-GLI-xxxxx-xxx.npz
    └── ...

mmsk/
└── dataset.json            # Metadata (train/val paths)
```

## 🔧 File yang Diupdate

1. **`prepare_dataset_no_split.py`** ✅ (BARU)
   - Tidak melakukan split
   - Hanya collect paths dari preprocessed_train dan preprocessed_val
   - Generate dataset.json

2. **`preprocessing_mmsk.ipynb`** 
   - Jalankan 2x:
     - 1x untuk training_data1_v2 → preprocessed_train
     - 1x untuk validation_data → preprocessed_val

## ⚠️ Perbedaan dengan prepare_dataset.py Lama

| Aspek | LAMA (prepare_dataset.py) | BARU (prepare_dataset_no_split.py) |
|-------|---------------------------|-------------------------------------|
| Split | ✅ Split 70/15/15 | ❌ Tidak split |
| Input | Raw .nii.gz files | ✅ Preprocessed .npz files |
| Output | dataset.json dengan split | ✅ dataset.json tanpa split |
| Alasan | Tidak tahu BraTS sudah split | ✅ Memanfaatkan split BraTS |

## 🚀 Quick Start

```bash
# 1. Preprocess training data
cd /home/mci/arsyadl/mmsk
jupyter notebook preprocessing_mmsk.ipynb
# Set: DATA_DIR = training_data1_v2, OUTPUT_DIR = preprocessed_train
# Run all cells

# 2. Preprocess validation data
# Set: DATA_DIR = validation_data, OUTPUT_DIR = preprocessed_val
# Run all cells

# 3. Generate dataset.json
python3 prepare_dataset_no_split.py

# 4. Training
python3 train_proposed_mmsk_tversky.py
```

## 💡 Keuntungan Approach Ini

1. ✅ **Tidak ada data leakage** - Split sudah dilakukan BraTS
2. ✅ **Konsisten dengan paper** - Menggunakan official split
3. ✅ **Reproducible** - Semua orang pakai split yang sama
4. ✅ **Lebih simple** - Tidak perlu logic split di code

## 📊 Verifikasi

```python
import json

# Load dataset.json
with open('/home/mci/arsyadl/mmsk/dataset.json') as f:
    data = json.load(f)

print(f"Training samples: {len(data['train'])}")
print(f"Validation samples: {len(data['val'])}")
print(f"Total: {len(data['train']) + len(data['val'])}")
```

## ❓ FAQ

**Q: Kenapa tidak pakai prepare_dataset.py yang lama?**
A: File lama melakukan split 70/15/15, padahal BraTS sudah menyediakan split terpisah.

**Q: Apakah perlu test set?**
A: Untuk skripsi, biasanya cukup train/val. Test bisa pakai validation_data atau subset dari training.

**Q: Bagaimana jika ingin test set?**
A: Bisa split manual dari training_data atau gunakan validation_data sebagai test.
