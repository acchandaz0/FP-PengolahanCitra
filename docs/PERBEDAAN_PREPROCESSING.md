# 🔄 Perbedaan Preprocessing Lama vs Baru

## 📊 Tabel Perbandingan Cepat

| No | Aspek | LAMA | BARU | Status |
|----|-------|------|------|--------|
| 1 | **Split Data** | ✅ Ada | ❌ Tidak ada | ⭐ BERUBAH |
| 2 | **Output Format** | ❓ Tidak jelas | ✅ .npz | ⭐ BERUBAH |
| 3 | **Modality Order** | ❌ t1c,t1n,t2f,t2w | ✅ t1n,t2w,t2f,t1c | ⭐ BERUBAH |
| 4 | **Target Size** | ❓ Tidak fix | ✅ 128×128×128 | ⭐ BERUBAH |
| 5 | **Normalization** | ✅ Z-score | ✅ Z-score | ✓ SAMA |
| 6 | **Cropping** | ✅ Non-zero | ✅ Non-zero | ✓ SAMA |
| 7 | **Resize Method** | ❓ Tidak jelas | ✅ ndimage.zoom | ⭐ BERUBAH |
| 8 | **Struktur Kode** | ❌ Exploratory | ✅ Modular | ⭐ BERUBAH |
| 9 | **Verifikasi** | ❌ Minimal | ✅ Lengkap | ⭐ BERUBAH |
| 10 | **Dokumentasi** | ❌ Minimal | ✅ Lengkap | ⭐ BERUBAH |

---

## ⭐ 5 Perubahan Paling Penting

### 1. **TIDAK ADA SPLIT DATA** 🔥
```
LAMA: Split train/val/test di preprocessing
BARU: Tidak ada split, dilakukan di prepare_dataset.py

ALASAN:
✓ BraTS sudah menyediakan training data terpisah
✓ Preprocessing fokus transformasi saja
✓ Lebih fleksibel untuk eksperimen
```

### 2. **FORMAT OUTPUT JELAS** 🔥
```python
LAMA: Format tidak jelas/tidak konsisten

BARU: .npz dengan struktur:
{
    'images': [4, 128, 128, 128],  # float32
    'seg': [128, 128, 128]         # uint8
}

ALASAN:
✓ 100% compatible dengan mmsk_3d_unet.py
✓ Format standar NumPy
✓ Compressed untuk hemat storage
```

### 3. **URUTAN MODALITY BENAR** 🔥
```python
LAMA: ['t1c', 't1n', 't2f', 't2w']  # ❌ Salah urutan

BARU: ['t1n', 't2w', 't2f', 't1c']  # ✅ Benar
       Channel 0: T1
       Channel 1: T2
       Channel 2: FLAIR
       Channel 3: T1ce

ALASAN:
✓ Sesuai ekspektasi model (line 238 mmsk_3d_unet.py)
✓ Urutan standar BraTS
✓ Konsisten dengan paper
```

### 4. **TARGET SIZE FIX** 🔥
```
LAMA: Tidak fix, bervariasi per sample

BARU: 128×128×128 (fixed untuk semua sample)

ALASAN:
✓ Sesuai paper: "Patch Extraction (128×128×128)"
✓ Konsisten untuk batch processing
✓ Optimal untuk GPU memory
```

### 5. **STRUKTUR KODE MODULAR** 🔥
```python
LAMA: Exploratory, tidak modular

BARU: Function-based
def preprocess_sample(folder_path):
    # Load → Normalize → Crop → Resize → Save
    return success, message

ALASAN:
✓ Mudah di-debug
✓ Error handling lebih baik
✓ Reusable
```

---

## 🎯 Yang SAMA (Tidak Berubah)

1. ✅ **Z-score Normalization** - Per modality, non-zero voxels
2. ✅ **Crop Non-zero Region** - Dengan margin
3. ✅ **Basic Preprocessing Steps** - Load → Normalize → Crop → Resize

---

## 💡 Kenapa Harus Berubah?

```
MASALAH LAMA:
├─ Output format tidak jelas
├─ Urutan modality salah
├─ Tidak compatible dengan model
└─ Sulit di-maintain

        ↓

SOLUSI BARU:
├─ Format jelas (.npz)
├─ Urutan modality benar
├─ 100% compatible dengan MMSK
└─ Production-ready

        ↓

HASIL:
✓ Langsung bisa training tanpa modifikasi!
```

---

## 🚨 Jika Sudah Preprocess dengan Notebook Lama

### ❌ **TIDAK BISA** langsung dipakai karena:
1. Format output tidak sesuai
2. Urutan modality salah
3. Target size tidak konsisten

### ✅ **SOLUSI:**
```bash
# Preprocess ulang dengan notebook baru
cd /home/mci/arsyadl/mmsk
jupyter notebook preprocessing_mmsk.ipynb
```

**Rekomendasi: PREPROCESS ULANG!** ✅

---

## 📝 Ringkasan Singkat

| Kategori | Perubahan |
|----------|-----------|
| **Critical** | Split data, Format output, Modality order |
| **Important** | Target size, Resize method, Struktur kode |
| **Nice to have** | Verifikasi, Dokumentasi |
| **Unchanged** | Normalization, Cropping |

**Total Perubahan Signifikan: 7 dari 10 aspek** 🔥

---

## 🎓 Kesimpulan

**Preprocessing BARU** adalah versi **production-ready** yang:
- ✅ 100% compatible dengan model MMSK
- ✅ Format output jelas dan konsisten
- ✅ Urutan modality benar
- ✅ Mudah di-maintain dan di-debug
- ✅ Siap langsung untuk training

**Preprocessing LAMA** adalah versi **exploratory** yang:
- ❌ Format output tidak jelas
- ❌ Urutan modality tidak konsisten
- ❌ Tidak compatible dengan model
- ❌ Sulit di-maintain

**Rekomendasi: Gunakan preprocessing BARU!** ⭐
