"""Pemuatan model, inferensi 3D, caching per-kasus, Dice, dan pemilihan irisan.

Semua hasil di-precompute sekali (lihat warmup) agar slider di UI mulus,
tidak ada pemanggilan model saat slider digeser.
"""
import numpy as np

import config

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

# nama_kasus -> dict(images, seg, pred, dice)
_CACHE = {}
_MODEL = None


def _build_model():
    """Bangun arsitektur model kosong.

    >>> GANTI isi fungsi ini sesuai definisi model-mu <<<
    Contoh:
        from models.mmsk_unet import MMSK3DUNet
        return MMSK3DUNet(in_channels=4, num_classes=5)   # sesuaikan argumen
    """
    raise NotImplementedError(
        "Isi _build_model() di inference.py: impor kelas MMSK3DUNet-mu "
        "dan kembalikan instance-nya dengan argumen yang benar."
    )


def load_model():
    """Muat model sekali (lazy). Return None bila CHECKPOINT_PATH belum di-set."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not _TORCH or config.CHECKPOINT_PATH is None:
        return None

    model = _build_model()
    ckpt = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(config.DEVICE).eval()   # eval() WAJIB: matikan dropout, pakai running stats
    _MODEL = model
    return _MODEL


def _predict_volume(images):
    """images: (4,128,128,128) float32  ->  label map (128,128,128) uint8.

    Asumsi default: model multiclass, output logits (1,C,128,128,128),
    argmax channel menghasilkan label {0..4} yang sama formatnya dengan 'seg'.
    """
    model = load_model()
    if model is None:
        return None
    x = torch.from_numpy(images).unsqueeze(0).to(config.DEVICE)   # (1,4,D,H,W)
    with torch.inference_mode():
        logits = model(x)                                        # (1,C,D,H,W)
    pred = logits.argmax(1).squeeze(0).to("cpu").numpy().astype(np.uint8)
    # --- Kalau model-mu MULTILABEL (3 channel sigmoid WT/TC/ET): jangan argmax.
    #     Lakukan threshold per channel, lalu rekonstruksi label map dari region. ---
    return pred


def _dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return 2.0 * np.logical_and(a, b).sum() / denom


def _region_dice(seg, pred):
    return {region: _dice(np.isin(seg, labels), np.isin(pred, labels))
            for region, labels in config.REGION_LABELS.items()}


def best_slice(seg, plane):
    """Indeks irisan dengan voxel tumor terbanyak pada bidang tertentu.
    Dipakai untuk mereset slider ke posisi paling informatif saat ganti plane/kasus."""
    tumor = seg > 0
    axis = {"Aksial": (1, 2), "Koronal": (0, 2), "Sagital": (0, 1)}[plane]
    return int(tumor.sum(axis=axis).argmax())


def get_case(case_name):
    """Muat + (bila model tersedia) prediksi sebuah kasus, dengan caching."""
    if case_name in _CACHE:
        return _CACHE[case_name]

    path = config.SAMPLE_CASES[case_name]
    data = np.load(path)
    images = data["images"].astype(np.float32)   # (4,128,128,128)
    seg = data["seg"].astype(np.uint8)           # (128,128,128)

    pred = _predict_volume(images)               # None bila model belum di-set
    dice = _region_dice(seg, pred) if pred is not None else None

    entry = {"images": images, "seg": seg, "pred": pred, "dice": dice}
    _CACHE[case_name] = entry
    return entry


def warmup():
    """Precompute semua kasus di awal agar interaksi UI instan."""
    for name in config.SAMPLE_CASES:
        get_case(name)
