╔══════════════════════════════════════════════════════════════════════════════╗
║           PERBANDINGAN PREPROCESSING LAMA VS BARU                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. SPLIT DATA                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

   LAMA (preprocessing_blomfix_fixed.ipynb):
   ├─ ✅ Ada split train/val/test
   ├─ Split dilakukan di preprocessing
   └─ Ratio: 70/10/20 (kemungkinan)
   
   BARU (preprocessing_mmsk.ipynb):
   ├─ ❌ TIDAK ada split
   ├─ Hanya preprocess semua data
   └─ Split dilakukan di prepare_dataset.py (saat training)
   
   ALASAN PERUBAHAN:
   ✓ BraTS sudah menyediakan training data terpisah
   ✓ Preprocessing fokus pada transformasi data saja
   ✓ Lebih fleksibel untuk eksperimen dengan split ratio berbeda

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. FORMAT OUTPUT                                                            │
└─────────────────────────────────────────────────────────────────────────────┘

   LAMA:
   ├─ Format: Tidak jelas/tidak konsisten
   ├─ Kemungkinan: Multiple files atau format custom
   └─ Tidak ada dokumentasi format yang jelas
   
   BARU:
   ├─ Format: .npz (NumPy compressed)
   ├─ Keys: 'images' dan 'seg'
   ├─ images: [4, D, H, W] dtype=float32
   ├─ seg: [D, H, W] dtype=uint8
   └─ ✓ Sesuai dengan input model MMSK 3D U-Net
   
   ALASAN PERUBAHAN:
   ✓ Kompatibilitas dengan mmsk_3d_unet.py
   ✓ Format standar dan mudah di-load
   ✓ Compressed untuk hemat storage

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. URUTAN MODALITY                                                          │
└─────────────────────────────────────────────────────────────────────────────┘

   LAMA:
   ├─ Urutan: ['t1c', 't1n', 't2f', 't2w']
   └─ Tidak konsisten dengan kebutuhan model
   
   BARU:
   ├─ Urutan: ['t1n', 't2w', 't2f', 't1c']
   ├─ Channel 0: T1 (t1n)
   ├─ Channel 1: T2 (t2w)
   ├─ Channel 2: FLAIR (t2f)
   └─ Channel 3: T1ce (t1c)
   
   ALASAN PERUBAHAN:
   ✓ Sesuai dengan ekspektasi model (line 238 mmsk_3d_unet.py)
   ✓ Urutan standar: T1, T2, FLAIR, T1ce
   ✓ Konsisten dengan paper BraTS

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. TARGET SIZE                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

   LAMA:
   ├─ Target size: Tidak fix/tidak jelas
   └─ Kemungkinan: Original size atau crop saja
   
   BARU:
   ├─ Target size: 128×128×128 (fixed)
   └─ Sesuai dengan paper
   
   ALASAN PERUBAHAN:
   ✓ Sesuai paper: "Patch Extraction (128×128×128)"
   ✓ Konsisten untuk semua sample
   ✓ Optimal untuk GPU memory

┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. NORMALISASI                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

   LAMA:
   ├─ Z-score normalization
   └─ Per modality (non-zero voxels)
   
   BARU:
   ├─ Z-score normalization
   └─ Per modality (non-zero voxels)
   
   ALASAN: SAMA ✓
   ✓ Sesuai paper
   ✓ Best practice untuk MRI

┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. CROPPING                                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

   LAMA:
   ├─ Crop non-zero region
   └─ Dengan margin
   
   BARU:
   ├─ Crop non-zero region
   └─ Dengan margin (margin=5)
   
   ALASAN: SAMA ✓
   ✓ Menghilangkan background hitam
   ✓ Fokus pada brain region

┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. RESIZING METHOD                                                          │
└─────────────────────────────────────────────────────────────────────────────┘

   LAMA:
   ├─ Method: Tidak jelas
   └─ Kemungkinan: Padding atau simple resize
   
   BARU:
   ├─ Method: scipy.ndimage.zoom
   ├─ Images: order=1 (bilinear)
   └─ Segmentation: order=0 (nearest neighbor)
   
   ALASAN PERUBAHAN:
   ✓ Zoom lebih smooth untuk images
   ✓ Nearest neighbor preserve labels untuk segmentation
   ✓ Konsisten dan reproducible

┌─────────────────────────────────────────────────────────────────────────────┐
│ 8. STRUKTUR KODE                                                            │
└─────────────────────────────────────────────────────────────────────────────┘

   LAMA:
   ├─ Struktur: Exploratory
   ├─ Banyak visualisasi
   └─ Tidak modular
   
   BARU:
   ├─ Struktur: Modular
   ├─ Function-based
   ├─ preprocess_sample() untuk 1 sample
   └─ Mudah di-debug dan di-maintain
   
   ALASAN PERUBAHAN:
   ✓ Lebih clean dan maintainable
   ✓ Mudah untuk batch processing
   ✓ Error handling lebih baik

┌─────────────────────────────────────────────────────────────────────────────┐
│ 9. VERIFIKASI OUTPUT                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

   LAMA:
   ├─ Verifikasi: Minimal
   └─ Tidak ada check format
   
   BARU:
   ├─ Verifikasi: Lengkap
   ├─ Check shape, dtype, range
   ├─ Check modality order
   └─ Visualisasi output
   
   ALASAN PERUBAHAN:
   ✓ Ensure correctness sebelum training
   ✓ Catch errors early
   ✓ Quality assurance

┌─────────────────────────────────────────────────────────────────────────────┐
│ 10. DOKUMENTASI                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

   LAMA:
   ├─ Dokumentasi: Minimal
   └─ Tidak ada penjelasan format output
   
   BARU:
   ├─ Dokumentasi: Lengkap
   ├─ Markdown cells menjelaskan setiap step
   ├─ README_PREPROCESSING.md
   └─ Clear expected output format
   
   ALASAN PERUBAHAN:
   ✓ Reproducibility
   ✓ Mudah dipahami orang lain
   ✓ Self-documenting code

╔══════════════════════════════════════════════════════════════════════════════╗
║                           RINGKASAN PERUBAHAN                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────┬─────────────────────┬─────────────────────────┐
│ Aspek                    │ LAMA                │ BARU                    │
├──────────────────────────┼─────────────────────┼─────────────────────────┤
│ Split Data               │ ✅ Ada              │ ❌ Tidak ada            │
│ Output Format            │ ❓ Tidak jelas      │ ✅ .npz (clear)         │
│ Modality Order           │ ❌ Tidak konsisten  │ ✅ T1,T2,FLAIR,T1ce     │
│ Target Size              │ ❓ Tidak fix        │ ✅ 128×128×128          │
│ Normalization            │ ✅ Z-score          │ ✅ Z-score              │
│ Cropping                 │ ✅ Non-zero         │ ✅ Non-zero             │
│ Resize Method            │ ❓ Tidak jelas      │ ✅ scipy.ndimage.zoom   │
│ Struktur Kode            │ ❌ Exploratory      │ ✅ Modular              │
│ Verifikasi               │ ❌ Minimal          │ ✅ Lengkap              │
│ Dokumentasi              │ ❌ Minimal          │ ✅ Lengkap              │
│ Kompatibilitas Model     │ ❌ Tidak jelas      │ ✅ 100% compatible      │
└──────────────────────────┴─────────────────────┴─────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║                        PERUBAHAN PALING PENTING                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. ⭐ TIDAK ADA SPLIT DATA
   └─ BraTS sudah split, preprocessing fokus transformasi saja

2. ⭐ FORMAT OUTPUT JELAS
   └─ .npz dengan keys 'images' [4,D,H,W] dan 'seg' [D,H,W]

3. ⭐ URUTAN MODALITY BENAR
   └─ T1, T2, FLAIR, T1ce (sesuai model)

4. ⭐ TARGET SIZE FIX
   └─ 128×128×128 (sesuai paper)

5. ⭐ KOMPATIBILITAS 100%
   └─ Output langsung bisa dipakai untuk training MMSK

╔══════════════════════════════════════════════════════════════════════════════╗
║                              KENAPA BERUBAH?                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

LAMA: Preprocessing exploratory, tidak jelas output format
      ↓
MASALAH: Output tidak compatible dengan model MMSK 3D U-Net
      ↓
BARU: Preprocessing production-ready, format jelas, 100% compatible
      ↓
HASIL: Siap langsung untuk training tanpa modifikasi!

╔══════════════════════════════════════════════════════════════════════════════╗
║                            MIGRATION PATH                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Jika sudah preprocess dengan notebook LAMA:
1. ❌ Output tidak bisa langsung dipakai
2. ✅ Harus preprocess ulang dengan notebook BARU
3. ✅ Atau convert format (tapi risky)

Rekomendasi: PREPROCESS ULANG dengan notebook baru! ✅
