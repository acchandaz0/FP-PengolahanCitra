# Preprocessing Notebook untuk MMSK 3D U-Net

## 📍 Lokasi
`/home/mci/arsyadl/mmsk/preprocessing_mmsk.ipynb`

## ✨ Fitur Utama

### 1. **Tidak Ada Split Data** ✅
- BraTS sudah menyediakan training data terpisah
- Preprocessing hanya fokus pada transformasi data
- Split dilakukan di `prepare_dataset.py`

### 2. **Format Output Sesuai Model** ✅
```python
# Output: .npz file
{
    'images': [4, 128, 128, 128],  # float32
    'seg': [128, 128, 128]         # uint8
}
```

### 3. **Urutan Modality yang Benar** ✅
```python
Channel 0: T1    (t1n)
Channel 1: T2    (t2w)
Channel 2: FLAIR (t2f)
Channel 3: T1ce  (t1c)
```

## 🔄 Preprocessing Steps

1. **Load 4 modalities** dalam urutan yang benar
2. **Z-score normalization** per modality (non-zero voxels)
3. **Crop non-zero region** dengan margin
4. **Resize to 128×128×128** menggunakan zoom
5. **Save as .npz** dengan format yang benar

## 📊 Perbedaan dengan Notebook Lama

| Aspek | Notebook Lama | Notebook Baru (MMSK) |
|-------|---------------|----------------------|
| Split data | ✅ Ada | ❌ Tidak ada |
| Output format | Tidak jelas | ✅ .npz dengan keys 'images', 'seg' |
| Modality order | Tidak konsisten | ✅ T1, T2, FLAIR, T1ce |
| Target size | Tidak fix | ✅ 128×128×128 |
| Normalization | Ada | ✅ Z-score per modality |

## 🚀 Cara Menggunakan

### 1. Buka Notebook
```bash
cd /home/mci/arsyadl/mmsk
jupyter notebook preprocessing_mmsk.ipynb
```

### 2. Jalankan Semua Cell
- Cell 1-2: Import libraries & configuration
- Cell 3-4: Explore dataset
- Cell 5-6: Test preprocessing
- Cell 7: Preprocess all data
- Cell 8-9: Verify output

### 3. Output
```
braTS/preprocessed/
├── BraTS-GLI-00001-000.npz
├── BraTS-GLI-00001-001.npz
└── ...
```

## ✅ Verifikasi Output

```python
import numpy as np

# Load sample
data = np.load('braTS/preprocessed/BraTS-GLI-00001-000.npz')

# Check format
assert data['images'].shape == (4, 128, 128, 128)
assert data['seg'].shape == (128, 128, 128)
assert data['images'].dtype == np.float32
assert data['seg'].dtype == np.uint8

# Check modality order
print("Channel 0 (T1):", data['images'][0].mean())
print("Channel 1 (T2):", data['images'][1].mean())
print("Channel 2 (FLAIR):", data['images'][2].mean())
print("Channel 3 (T1ce):", data['images'][3].mean())
```

## 🎯 Next Steps

Setelah preprocessing selesai:

```bash
cd mmsk
python3 prepare_dataset.py      # Split train/val/test
python3 train_proposed_mmsk_tversky.py  # Training
```

## 💡 Tips

1. **Jalankan cell satu per satu** untuk debugging
2. **Visualisasi sample** sebelum preprocess semua data
3. **Verifikasi output** sebelum training
4. **Backup data** jika perlu

## ⚠️ Catatan Penting

- **Modality order HARUS benar**: T1, T2, FLAIR, T1ce
- **Tidak ada split** di preprocessing
- **Output format** harus sesuai dengan model
- **Z-score normalization** hanya pada non-zero voxels
