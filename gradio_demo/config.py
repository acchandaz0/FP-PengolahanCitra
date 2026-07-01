"""Konfigurasi demo — cukup ubah path & konstanta di file ini."""

# === 1. PATH YANG HARUS KAMU ISI ==========================================
# Checkpoint bobot terbaik hasil training.
# Biarkan None dulu kalau mau menguji UI tanpa model (hanya ground truth tampil).
CHECKPOINT_PATH = None
# contoh: CHECKPOINT_PATH = "/home/mci/arsyadl/mmsk/checkpoints/best_model.pth"

# Kasus contoh untuk dropdown:  "Nama yang tampil" -> path file .npz
SAMPLE_CASES = {
    "Kasus 1 — seimbang (02303)": "/home/mci/arsyadl/braTS/preprocessed/BraTS-GLI-02303-100.npz",
    # tambah kasus lain bila perlu:
    # "Kasus 2 — ET besar":        "/home/mci/arsyadl/braTS/preprocessed/BraTS-GLI-XXXXX-100.npz",
    # "Kasus 3 — tumor kecil":     "/home/mci/arsyadl/braTS/preprocessed/BraTS-GLI-XXXXX-100.npz",
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
