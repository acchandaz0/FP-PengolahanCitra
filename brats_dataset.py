"""
brats_dataset.py — Shared dataset, validation, and logging utilities
for all ablation study variants (Baseline 1/2, Ablation 1/2, Proposed).

BraTS 2024 Synapse GLI label convention (RAW, no remap):
    0 = Background
    1 = NCR  (Non-Enhancing Tumor Core / Necrosis)
    2 = ED   (Peritumoral Edema)
    3 = NET  (Non-Enhancing Tumor)
    4 = ET   (Enhancing Tumor)

All variants use out_channels=5.
DiceMetric(include_background=False) → reports 4 foreground classes: NCR/ED/NET/ET.

Confirmed by running diagnose_labels.py on actual preprocessed .npz files:
    seg dtype  : uint8
    seg shape  : (128, 128, 128)   — no channel dim, unsqueeze added in _load()
    images shape: (4, 128, 128, 128)
    unique labels across 30 files: {0, 1, 2, 3, 4}
"""

import torch
import numpy as np
import warnings
import zlib
import logging
import sys
from pathlib import Path
from torch.utils.data import Dataset


# ── Label convention ──────────────────────────────────────────────────────────
VALID_RAW_LABELS  = {0, 1, 2, 3, 4}
NUM_CLASSES       = 5          # including background
NUM_FG_CLASSES    = 4          # NCR, ED, NET, ET
FG_CLASS_NAMES    = ["NCR", "ED", "NET", "ET"]
OUT_CHANNELS      = 5


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logger(name: str, output_dir: Path) -> logging.Logger:
    """
    Structured logger: INFO → stdout, DEBUG → <output_dir>/train.log.
    Safe to call multiple times (idempotent via handler check).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Log file: {log_path}")
    return logger


# ── Pre-flight validation ─────────────────────────────────────────────────────
def validate_npz_files(
    file_list: list[str],
    logger: logging.Logger,
    verbose: bool = True,
) -> list[str]:
    """
    Scan every .npz, force-decompress both arrays, validate shapes and labels.
    Returns only healthy files; corrupted ones are logged and dropped.

    Validation criteria:
        images : float32, shape (4, D, H, W)
        seg    : any int dtype, shape (D, H, W) — 3-D, no channel dim
        labels : all values must be in {0, 1, 2, 3, 4}
    """
    clean, bad = [], []
    logger.info(f"[Validation] Scanning {len(file_list)} .npz files ...")

    for i, fp in enumerate(file_list):
        try:
            with np.load(fp) as npz:
                img = npz["images"]
                seg = npz["seg"]
                _   = img.sum()          # force full decompression
                _   = seg.sum()

                if img.ndim != 4 or img.shape[0] != 4:
                    raise ValueError(
                        f"images shape {img.shape} invalid; expected (4,D,H,W)"
                    )
                if seg.ndim != 3:
                    raise ValueError(
                        f"seg shape {seg.shape} invalid; expected (D,H,W)"
                    )
                invalid = set(np.unique(seg).tolist()) - VALID_RAW_LABELS
                if invalid:
                    raise ValueError(
                        f"Unexpected label values {sorted(invalid)} "
                        f"(valid: {{0,1,2,3,4}})"
                    )

            clean.append(fp)
            if verbose and i % 100 == 0:
                logger.debug(f"  [{i+1}/{len(file_list)}] OK: {Path(fp).name}")

        except (zlib.error, EOFError, OSError, KeyError, ValueError) as e:
            bad.append((fp, str(e)))
            logger.warning(f"[CORRUPT] {Path(fp).name} → {e}")

    logger.info(
        f"[Validation] Done. {len(clean)} healthy, {len(bad)} dropped."
    )
    if bad:
        logger.warning("[Validation] Corrupt/invalid files:")
        for fp, reason in bad:
            logger.warning(f"  ✗ {fp}  ({reason})")

    return clean


# ── Dataset ───────────────────────────────────────────────────────────────────
class BraTSNpzDataset(Dataset):
    """
    Load nnU-Net-style preprocessed BraTS 2024 .npz patches.

    .npz contents:
        images : float32, (4, 128, 128, 128)  — T1, T2, FLAIR, T1ce
                 Already z-score normalised per modality (nonzero voxels)
                 by the nnU-Net preprocessor — no further scaling needed.
        seg    : uint8,   (128, 128, 128)     — raw labels {0,1,2,3,4}
                 0=BG, 1=NCR, 2=ED, 3=NET, 4=ET
                 No remap required; use directly as class indices.

    __getitem__ returns:
        {
            "image" : FloatTensor  (4, D, H, W)
            "label" : LongTensor   (1, D, H, W)   <- unsqueeze(0) added
        }

    Augmentation (train=True):
        Random flip along each spatial axis (p=0.5 each)
        Random intensity shift ±0.1 on image (p=0.5), clamped to [0,1]

    Resilience:
        Any runtime load failure returns a zero-filled tensor of the last
        seen shape instead of crashing the DataLoader worker.
    """

    _fallback_image_shape = (4, 128, 128, 128)
    _fallback_seg_shape   = (1, 128, 128, 128)

    def __init__(self, file_list: list[str], augment: bool = False):
        self.files   = file_list
        self.augment = augment
        self._log    = logging.getLogger("brats_dataset")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._load(idx)
        except Exception as e:
            self._log.warning(
                f"Runtime load failed [{idx}] {self.files[idx]}: {e}. "
                "Returning zero-filled fallback."
            )
            warnings.warn(
                f"[BraTSNpzDataset] Failed to load {self.files[idx]}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return {
                "image": torch.zeros(self._fallback_image_shape, dtype=torch.float32),
                "label": torch.zeros(self._fallback_seg_shape,   dtype=torch.int64),
            }

    def _load(self, idx: int) -> dict:
        npz = np.load(self.files[idx])

        # images: already normalised by nnU-Net preprocessor
        image = torch.from_numpy(npz["images"].astype(np.float32))   # (4,D,H,W)

        # seg: raw uint8 labels {0,1,2,3,4}, add channel dim
        seg = torch.from_numpy(npz["seg"].astype(np.int64)).unsqueeze(0)  # (1,D,H,W)

        # Defensive clamp — should be a no-op on healthy data,
        # but prevents a corrupt patch from causing a loss NaN
        seg = seg.clamp(0, 4)

        # Update fallback shapes from real data
        BraTSNpzDataset._fallback_image_shape = tuple(image.shape)
        BraTSNpzDataset._fallback_seg_shape   = tuple(seg.shape)

        if self.augment:
            for axis in [1, 2, 3]:
                if torch.rand(1).item() > 0.5:
                    image = torch.flip(image, [axis])
                    seg   = torch.flip(seg,   [axis])
            if torch.rand(1).item() > 0.5:
                shift = torch.rand(1).item() * 0.2 - 0.1    # uniform in [-0.1, 0.1]
                image = (image + shift).clamp(0.0, 1.0)

        return {"image": image, "label": seg}