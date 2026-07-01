"""Pengambilan irisan per-bidang (plane) dan render overlay berwarna.

Semua fungsi render menerima irisan 2D dan mengembalikan numpy uint8 RGB
(siap untuk gr.Image).
"""
import numpy as np

import config

PLANES = ["Aksial", "Koronal", "Sagital"]

# Berapa kali np.rot90 diterapkan agar kepala tampak tegak di tiap bidang.
# Cek dengan mata sekali saat pertama jalan; ubah angkanya (0-3) bila orientasinya
# terasa miring/terbalik. Ini hanya soal tampilan, tidak mengubah kebenaran data.
PLANE_ROT = {"Aksial": 1, "Koronal": 1, "Sagital": 1}


def plane_size(volume, plane):
    """Jumlah irisan (panjang slider) untuk bidang tertentu."""
    return {"Aksial": volume.shape[0],
            "Koronal": volume.shape[1],
            "Sagital": volume.shape[2]}[plane]


def take_slice(volume, plane, idx):
    """Ambil satu irisan 2D dari volume (128,128,128) pada bidang & indeks tertentu."""
    if plane == "Aksial":
        s = volume[idx, :, :]
    elif plane == "Koronal":
        s = volume[:, idx, :]
    else:  # Sagital
        s = volume[:, :, idx]
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
    return _grayscale_base(img_slice).clip(0, 255).astype(np.uint8)


def overlay_per_label(img_slice, seg_slice, alpha=config.OVERLAY_ALPHA):
    """Warnai tiap label mentah (NCR/ED/NET/ET) di atas grayscale."""
    rgb = _grayscale_base(img_slice)
    for lab, color in config.LABEL_COLORS.items():
        _blend(rgb, seg_slice == lab, color, alpha)
    return rgb.clip(0, 255).astype(np.uint8)


def overlay_per_region(img_slice, seg_slice, alpha=config.OVERLAY_ALPHA):
    """Warnai region komposit WT/TC/ET (digambar WT -> TC -> ET, ET menimpa di atas)."""
    rgb = _grayscale_base(img_slice)
    for region in ["WT", "TC", "ET"]:
        mask = np.isin(seg_slice, config.REGION_LABELS[region])
        _blend(rgb, mask, config.REGION_COLORS[region], alpha)
    return rgb.clip(0, 255).astype(np.uint8)
