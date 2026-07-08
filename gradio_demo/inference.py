"""Pemuatan model, inferensi 3D, caching per-kasus, Dice, dan pemilihan irisan.

Semua hasil di-precompute sekali (lihat warmup) agar slider di UI mulus,
tidak ada pemanggilan model saat slider digeser.
"""
import os
import sys

import numpy as np

import config

# Agar `mmsk_3d_unet.py` di folder induk (mmsk/) bisa diimpor saat menjalankan
# `python app.py` dari dalam gradio_demo/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

# nama_kasus -> dict(images, seg)   (independen model, dimuat sekali)
_DATA_CACHE = {}
# (nama_model, nama_kasus) -> dict(pred, dice)   (bergantung model)
_PRED_CACHE = {}
# nama_model -> objek model ter-load (atau None bila gagal/belum di-set)
_MODELS = {}


def _build_model(model_name):
    """Bangun arsitektur model kosong untuk `model_name`.

    Saat ini semua model memakai arsitektur Proposed (MMSK-3D U-Net) dengan
    out_channels=5 (BG/NCR/ED/NET/ET), argmax-nya cocok dengan format 'seg'.

    NB: pakai `mmsk_3d_unet_old` — checkpoint proposed_best_model.pth dilatih
    dengan varian modal_gate GAP+Linear (versi lama), bukan mmsk_3d_unet.py
    yang sudah di-refactor ke gating spasial Conv3d (bobotnya tidak cocok).

    >>> Kalau model pembanding punya ARSITEKTUR BERBEDA, cabangkan di sini
        berdasarkan `model_name` dan kembalikan modul yang sesuai. <<<
    """
    from mmsk_3d_unet_old import MMSK3DUNet
    return MMSK3DUNet(in_channels=4, out_channels=5)


def load_model(model_name):
    """Muat model bernama `model_name` sekali (lazy, dengan caching).

    Return None (tanpa meng-crash UI) bila: torch tak tersedia, path None,
    file checkpoint belum ada (masih placeholder), atau bobot gagal dimuat.
    """
    if model_name in _MODELS:
        return _MODELS[model_name]

    path = config.MODELS.get(model_name)
    model = None
    if not _TORCH:
        pass
    elif path is None:
        pass
    elif not os.path.exists(path):
        print(f"[inference] Checkpoint '{model_name}' belum ada: {path} — prediksi dilewati.")
    else:
        try:
            model = _build_model(model_name)
            ckpt = torch.load(path, map_location=config.DEVICE)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
            model.to(config.DEVICE).eval()   # eval() WAJIB: matikan dropout, pakai running stats
        except Exception as e:  # arsitektur tak cocok / file rusak -> jangan crash UI
            print(f"[inference] Gagal memuat '{model_name}' dari {path}: {e}")
            model = None

    _MODELS[model_name] = model
    return model


def _predict_volume(images, model_name):
    """images: (4,128,128,128) float32  ->  label map (128,128,128) uint8.

    Asumsi default: model multiclass, output logits (1,C,128,128,128),
    argmax channel menghasilkan label {0..4} yang sama formatnya dengan 'seg'.
    """
    model = load_model(model_name)
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
    import visualization as vis
    tumor = seg > 0
    slice_axis = vis.PLANE_AXIS[plane]
    other = tuple(a for a in (0, 1, 2) if a != slice_axis)   # jumlahkan 2 sumbu lain
    return int(tumor.sum(axis=other).argmax())


def _load_data(case_name):
    """Muat images + seg sebuah kasus (independen model), dengan caching."""
    if case_name in _DATA_CACHE:
        return _DATA_CACHE[case_name]
    path = config.SAMPLE_CASES[case_name]
    data = np.load(path)
    entry = {
        "images": data["images"].astype(np.float32),   # (4,128,128,128)
        "seg": data["seg"].astype(np.uint8),           # (128,128,128)
    }
    _DATA_CACHE[case_name] = entry
    return entry


def get_case(case_name, model_name):
    """Muat kasus + prediksi model terpilih. images/seg dan prediksi di-cache
    terpisah, jadi ganti model tak memuat ulang data, dan sebaliknya."""
    data = _load_data(case_name)

    key = (model_name, case_name)
    if key not in _PRED_CACHE:
        pred = _predict_volume(data["images"], model_name)   # None bila model belum di-set
        dice = _region_dice(data["seg"], pred) if pred is not None else None
        _PRED_CACHE[key] = {"pred": pred, "dice": dice}
    p = _PRED_CACHE[key]

    return {"images": data["images"], "seg": data["seg"], "pred": p["pred"], "dice": p["dice"]}


def warmup(model_name=None):
    """Precompute semua kasus untuk `model_name` (default: model awal) agar
    interaksi UI instan. Model lain dihitung lazy saat pertama dipilih."""
    if model_name is None:
        model_name = config.DEFAULT_MODEL
    for name in config.SAMPLE_CASES:
        get_case(name, model_name)
