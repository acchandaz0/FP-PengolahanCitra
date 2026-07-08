"""Konfigurasi demo — cukup ubah path & konstanta di file ini."""

# === 1. MODEL YANG DIBANDINGKAN ===========================================
# "Nama tampil di UI"  ->  path checkpoint (.pth) best model tiap model.
# Radio "Model" di UI mengambil kunci-kunci dict ini. Ganti path PLACEHOLDER
# di bawah dengan checkpoint asli tiap model (nanti kamu isi sendiri) dan
# ganti pula label-nya bila perlu.
# Set None kalau ingin sebuah model menampilkan ground truth saja (tanpa prediksi).
MODELS = {
    "Proposed — MMSK-3D U-Net": "/home/mci/arsyadl/mmsk/gradio_demo/proposed_best_model.pth",
    "Model 2":                  "/PLACEHOLDER/ganti/model2_best_model.pth",
    "Model 3":                  "/PLACEHOLDER/ganti/model3_best_model.pth",
    "Model 4":                  "/PLACEHOLDER/ganti/model4_best_model.pth",
    "Model 5":                  "/PLACEHOLDER/ganti/model5_best_model.pth",
}
# Model yang aktif saat pertama kali dibuka.
DEFAULT_MODEL = "Proposed — MMSK-3D U-Net"

# Kasus contoh untuk dropdown:  "Nama yang tampil" -> path file .npz
SAMPLE_CASES = {
    "Kasus 1 — seimbang (02303)": "/home/mci/arsyadl/braTS/preprocessed/BraTS-GLI-02303-100.npz",
    "Kasus 2 — ET besar (02793)": "/home/mci/arsyadl/braTS/preprocessed/BraTS-GLI-02793-100.npz",
    "Kasus 3 — tumor kecil (02798)": "/home/mci/arsyadl/braTS/preprocessed/BraTS-GLI-02798-100.npz",
}

# === 2. SKEMA LABEL & WARNA ===============================================
# Urutan channel pada array 'images'. WAJIB cocok dengan preprocessing-mu,
# kalau tertukar, latar grayscale jadi modalitas yang salah.
MODALITY_ORDER = ["T1", "T1ce", "T2", "FLAIR"]
DEFAULT_MODALITY = "T1ce"   # T1ce paling jelas menampilkan enhancing tumor

# Label mentah BraTS GLI: 0=BG, 1=NCR, 2=ED, 3=NET, 4=ET
LABEL_NAMES = {1: "NCR", 2: "ED", 3: "NET", 4: "ET"}
LABEL_COLORS = {            # RGB 0-255
    1: (255,  80,  80),     # NCR  - merah
    2: (255, 200,  60),     # ED   - kuning
    3: ( 80, 160, 255),     # NET  - biru
    4: ( 80, 220, 120),     # ET   - hijau
}

# Region komposit evaluasi. >>> VERIFIKASI dengan definisi di Bab III/IV-mu <<<
#   WT (Whole Tumor) = 1 + 2 + 3 + 4   (semua jaringan tumor)
#   TC (Tumor Core)  = 1 + 3 + 4       (semua kecuali edema)
#   ET (Enhancing)   = 4
REGION_LABELS = {
    "WT": [1, 2, 3, 4],
    "TC": [1, 3, 4],
    "ET": [4],
}
REGION_COLORS = {
    "WT": (255, 200,  60),
    "TC": ( 80, 160, 255),
    "ET": ( 80, 220, 120),
}

# === 3. PARAMETER TAMPILAN & RUNTIME ======================================
OVERLAY_ALPHA = 0.45        # 0 = transparan, 1 = warna penuh
DEVICE = "cuda"             # "cuda" atau "cpu"
