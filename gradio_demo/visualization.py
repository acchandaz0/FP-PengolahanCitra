"""Pengambilan irisan per-bidang (plane) dan render overlay berwarna.

Semua fungsi render menerima irisan 2D dan mengembalikan numpy uint8 RGB
(siap untuk gr.Image).
"""
import numpy as np

import config

PLANES = ["Aksial", "Koronal", "Sagital"]

# Sumbu volume (X,Y,Z)=(0,1,2) yang menjadi indeks irisan tiap bidang.
# Satu-satunya sumber kebenaran: take_slice, plane_size, dan best_slice
# (inference.py) semua mengambil dari sini agar tidak pernah terbalik lagi.
#   sumbu 0 -> Sagital,  sumbu 1 -> Koronal,  sumbu 2 -> Aksial
PLANE_AXIS = {"Sagital": 0, "Koronal": 1, "Aksial": 2}

# Berapa kali np.rot90 diterapkan agar kepala tampak tegak di tiap bidang.
# Cek dengan mata sekali saat pertama jalan; ubah angkanya (0-3) bila orientasinya
# terasa miring/terbalik. Ini hanya soal tampilan, tidak mengubah kebenaran data.
PLANE_ROT = {"Aksial": 1, "Koronal": 1, "Sagital": 1}

# Faktor perbesaran irisan sebelum dikirim ke gr.Image. Irisan aslinya kecil
# (128x128 px) sehingga tampil mungil di tengah panel. Dengan UPSCALE=4 jadi
# 512x512 px sehingga mengisi panel dengan tajam. Nearest-neighbor menjaga tepi
# label segmentasi tetap tegas (bukan blur).
UPSCALE = 4


def _upscale(rgb):
    """Perbesar RGB uint8 (H,W,3) dengan nearest-neighbor tanpa dependensi tambahan."""
    if UPSCALE <= 1:
        return rgb
    return np.repeat(np.repeat(rgb, UPSCALE, axis=0), UPSCALE, axis=1)


def plane_size(volume, plane):
    """Jumlah irisan (panjang slider) untuk bidang tertentu."""
    return volume.shape[PLANE_AXIS[plane]]


def take_slice(volume, plane, idx):
    """Ambil satu irisan 2D dari volume (128,128,128) pada bidang & indeks tertentu."""
    s = np.take(volume, idx, axis=PLANE_AXIS[plane])
    k = PLANE_ROT[plane]
    return np.rot90(s, k) if k else s


def _grayscale_base(img_slice):
    """Normalisasi irisan modalitas (float ternormalisasi) ke RGB grayscale 0-255."""
    lo, hi = float(img_slice.min()), float(img_slice.max())
    base = (img_slice - lo) / (hi - lo + 1e-8) * 255.0
    base = base.astype(np.uint8)
    return np.stack([base, base, base], axis=-1).astype(np.float32)


def _blend(rgb, mask, color, alpha):
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(color, dtype=np.float32)


def render_raw(img_slice):
    """Panel kiri: hanya grayscale, tanpa overlay."""
    return _upscale(_grayscale_base(img_slice).clip(0, 255).astype(np.uint8))


def overlay_per_label(img_slice, seg_slice, alpha=config.OVERLAY_ALPHA):
    """Warnai tiap label mentah (NCR/ED/NET/ET) di atas grayscale."""
    rgb = _grayscale_base(img_slice)
    for lab, color in config.LABEL_COLORS.items():
        _blend(rgb, seg_slice == lab, color, alpha)
    return _upscale(rgb.clip(0, 255).astype(np.uint8))


def overlay_per_region(img_slice, seg_slice, alpha=config.OVERLAY_ALPHA):
    """Warnai region komposit WT/TC/ET (digambar WT -> TC -> ET, ET menimpa di atas)."""
    rgb = _grayscale_base(img_slice)
    for region in ["WT", "TC", "ET"]:
        mask = np.isin(seg_slice, config.REGION_LABELS[region])
        _blend(rgb, mask, config.REGION_COLORS[region], alpha)
    return _upscale(rgb.clip(0, 255).astype(np.uint8))
