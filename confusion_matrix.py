#!/usr/bin/env python3
"""
confusion_matrix.py
-------------------
Membuat confusion matrix per-voxel (ternormalisasi per baris) dari sebuah
checkpoint, untuk menunjukkan KE MANA voxel kelas yang collapse "bocor".

Mengimpor modul pipeline-mu sendiri (brats_utils + kelas model) supaya
data loading & preprocessing IDENTIK dengan training/evaluasi lain —
ini menjaga konsistensi metodologi.

Pakai:
  python confusion_matrix.py \
      --variant proposed \
      --checkpoint path/ke/best_model.pth \
      --dataset_json path/ke/dataset.json \
      --gpu 0 --out cm/

Varian yang didukung (menentukan kelas model yang dipakai):
  baseline1, baseline2  -> MONAI UNet
  ablation1             -> BGSK3DUNet
  ablation2, proposed   -> MMSK3DUNet

Output:
  cm/<variant>_cm_counts.npy   : matriks 5x5 (hitungan voxel mentah)
  cm/<variant>_cm_norm.png     : confusion matrix ternormalisasi per baris
  cm/<variant>_cm_norm_fg.png  : versi 4x4 (tanpa background) untuk fokus tumor

CATATAN:
  - Background = indeks 0. Kelas: 0=BG, 1=NCR, 2=ED, 3=NET, 4=ET.
  - Normalisasi PER BARIS (per kelas ground-truth): tiap baris = "dari semua
    voxel yang SEBENARNYA kelas X, berapa porsi diprediksi jadi tiap kelas".
    Ini cara baca collapse: baris NCR yang sebagian besar jatuh ke kolom NET/ED
    = NCR di-misclassify jadi NET/ED.
  - Pakai best_model.pth. Skrip menangani dua format simpanan:
    dict {"model_state_dict": ...} ATAU raw state_dict (kasus Ablasi 1).
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Impor pipeline-mu (HARUS bisa di-import dari folder ini) ──────────
from brats_utils import (
    BraTSNpzDataset,
    validate_npz_files,
    OUT_CHANNELS,
)

CLASS_NAMES_FULL = ["BG", "NCR", "ED", "NET", "ET"]   # indeks 0..4
CLASS_NAMES_FG   = ["NCR", "ED", "NET", "ET"]          # indeks 1..4


def build_model(variant, device):
    """Bangun arsitektur sesuai varian (mirror dari script training)."""
    v = variant.lower()
    if v in ("baseline1", "baseline2", "b1", "b2"):
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3, in_channels=4, out_channels=OUT_CHANNELS,
            channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=2,
        )
    elif v in ("ablation1", "a1"):
        from bgsk_3d_unet_patched import BGSK3DUNet
        model = BGSK3DUNet(in_channels=4, out_channels=OUT_CHANNELS)
    elif v in ("ablation2", "a2", "proposed", "usulan"):
        from mmsk.mmsk_3d_unet_old import MMSK3DUNet
        model = MMSK3DUNet(in_channels=4, out_channels=OUT_CHANNELS,
                           store_attention=False)
    else:
        raise ValueError(f"Varian tidak dikenal: {variant}")
    return model.to(device)


def load_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        ep = ckpt.get("epoch")
        print(f"[i] Loaded model_state_dict (epoch={ep})")
    else:
        # raw state_dict (mis. best_model.pth Ablasi 1)
        model.load_state_dict(ckpt)
        print("[i] Loaded raw state_dict")
    return model


@torch.no_grad()
def accumulate_cm(model, loader, device, num_classes=5):
    """Akumulasi confusion matrix per-voxel via bincount (hemat memori)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    use_amp = device.type == "cuda"
    for i, batch in enumerate(loader):
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        labels = labels.long().clamp_(0, num_classes - 1)  # guard sama spt training

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(inputs)
        pred = logits.argmax(dim=1)  # [B,H,W,D]

        gt = labels.squeeze(1) if labels.dim() == pred.dim() + 1 else labels
        gt_flat = gt.reshape(-1).cpu().numpy()
        pr_flat = pred.reshape(-1).cpu().numpy()

        idx = gt_flat * num_classes + pr_flat
        binc = np.bincount(idx, minlength=num_classes * num_classes)
        cm += binc.reshape(num_classes, num_classes)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1} volume diproses")
    return cm


def plot_cm(cm, names, title, out_path):
    """Plot confusion matrix ternormalisasi per baris."""
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cmn = cm / row_sums

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    for r in range(len(names)):
        for c in range(len(names)):
            val = cmn[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Proporsi per baris")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[✓] {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    help="baseline1|baseline2|ablation1|ablation2|proposed")
    ap.add_argument("--checkpoint", required=True, help="path best_model.pth")
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="./cm")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")

    with open(args.dataset_json) as f:
        ds = json.load(f)
    val_files = validate_npz_files(ds["val"], logger=None) \
        if "val" in ds else []
    if not val_files:
        # validate_npz_files mungkin butuh logger; fallback tanpa validasi
        val_files = ds["val"]

    val_ds = BraTSNpzDataset(val_files, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = build_model(args.variant, device)
    model = load_weights(model, args.checkpoint, device)

    print(f"[i] Menghitung confusion matrix pada {len(val_files)} volume val ...")
    cm = accumulate_cm(model, val_loader, device, num_classes=OUT_CHANNELS)

    # simpan hitungan mentah
    np.save(Path(args.out) / f"{args.variant}_cm_counts.npy", cm)
    print(f"[✓] {args.out}/{args.variant}_cm_counts.npy")

    # plot full (5x5, dengan BG)
    plot_cm(cm, CLASS_NAMES_FULL,
            f"Confusion Matrix (ternormalisasi) — {args.variant}",
            Path(args.out) / f"{args.variant}_cm_norm.png")

    # plot foreground-only (4x4, buang BG row+col)
    cm_fg = cm[1:, 1:]
    plot_cm(cm_fg, CLASS_NAMES_FG,
            f"Confusion Matrix Foreground — {args.variant}",
            Path(args.out) / f"{args.variant}_cm_norm_fg.png")

    print("Selesai.")