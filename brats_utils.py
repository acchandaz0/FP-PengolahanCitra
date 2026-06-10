"""
brats_utils.py — Shared utilities for all ablation variants.

BraTS 2024 Synapse GLI label convention (RAW, confirmed by diagnose_labels.py):
    seg dtype : uint8,  shape (128,128,128) — no channel dim
    0=BG  1=NCR  2=ED  3=NET  4=ET
    out_channels = 5  (for all variants)

Metrics computed per-epoch in validate():
    DSC        — Dice Similarity Coefficient per foreground class
    HD95       — 95th-percentile Hausdorff Distance per class (mm)
    Sensitivity— True Positive Rate (Recall) per class
    Precision  — Positive Predictive Value per class

FLOPs:
    Computed once at startup via fvcore (if available) or thop fallback.
    Stored in results.json under "flops_giga".

All variants share:
    - BraTSNpzDataset  (identical augmentation)
    - validate_npz_files()
    - compute_metrics()
    - setup_logger()
    - OUT_CHANNELS = 5
    - FG_CLASS_NAMES = ["NCR", "ED", "NET", "ET"]
"""

import torch
import torch.nn.functional as F
import numpy as np
import warnings
import zlib
import zipfile
import logging
import sys
from pathlib import Path
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_RAW_LABELS = {0, 1, 2, 3, 4}
OUT_CHANNELS     = 5
FG_CLASS_NAMES   = ["NCR", "ED", "NET", "ET"]   # indices 1-4 after bg removal
NUM_FG           = 4


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logger(name: str, output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Log file: {output_dir / 'train.log'}")
    return logger


# ── Pre-flight validation ─────────────────────────────────────────────────────
def validate_npz_files(
    file_list: list[str],
    logger: logging.Logger,
    verbose: bool = True,
) -> list[str]:
    clean, bad = [], []
    logger.info(f"[Validation] Scanning {len(file_list)} .npz files ...")
    for i, fp in enumerate(file_list):
        try:
            with np.load(fp) as npz:
                img = npz["images"]
                seg = npz["seg"]
                _   = img.sum()
                _   = seg.sum()
                if img.ndim != 4 or img.shape[0] != 4:
                    raise ValueError(f"images shape {img.shape}; expected (4,D,H,W)")
                if seg.ndim != 3:
                    raise ValueError(f"seg shape {seg.shape}; expected (D,H,W)")
                invalid = set(np.unique(seg).tolist()) - VALID_RAW_LABELS
                if invalid:
                    raise ValueError(f"Unexpected label values {sorted(invalid)}")
            clean.append(fp)
            if verbose and i % 100 == 0:
                logger.debug(f"  [{i+1}/{len(file_list)}] OK: {Path(fp).name}")
        except (zlib.error, EOFError, OSError, KeyError, ValueError, zipfile.BadZipFile) as e:
            bad.append((fp, str(e)))
            logger.warning(f"[CORRUPT] {Path(fp).name} -> {e}")
    logger.info(f"[Validation] {len(clean)} healthy, {len(bad)} dropped.")
    for fp, reason in bad:
        logger.warning(f"  x {fp}  ({reason})")
    return clean


# ── Dataset ───────────────────────────────────────────────────────────────────
class BraTSNpzDataset(Dataset):
    """
    Raw BraTS 2024 .npz loader. No label remap — labels {0..4} used directly.
    seg shape on disk: (D,H,W) uint8 -> loaded as LongTensor (1,D,H,W).
    """
    _fallback_img = (4, 128, 128, 128)
    _fallback_seg = (1, 128, 128, 128)

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
            self._log.warning(f"Load failed [{idx}] {self.files[idx]}: {e}")
            warnings.warn(str(e), RuntimeWarning, stacklevel=2)
            return {
                "image": torch.zeros(self._fallback_img, dtype=torch.float32),
                "label": torch.zeros(self._fallback_seg, dtype=torch.int64),
            }

    def _load(self, idx: int) -> dict:
        npz   = np.load(self.files[idx])
        image = torch.from_numpy(npz["images"].astype(np.float32))        # (4,D,H,W)
        seg   = torch.from_numpy(npz["seg"].astype(np.int64)).unsqueeze(0) # (1,D,H,W)
        seg   = seg.clamp(0, 4)   # defensive only — no-op on healthy data

        BraTSNpzDataset._fallback_img = tuple(image.shape)
        BraTSNpzDataset._fallback_seg = tuple(seg.shape)

        if self.augment:
            for axis in [1, 2, 3]:
                if torch.rand(1).item() > 0.5:
                    image = torch.flip(image, [axis])
                    seg   = torch.flip(seg,   [axis])
            if torch.rand(1).item() > 0.5:
                image = (image + (torch.rand(1).item() * 0.2 - 0.1)).clamp(0., 1.)

        return {"image": image, "label": seg}


# ── Metrics ───────────────────────────────────────────────────────────────────
def _onehot(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert (B,1,D,H,W) int label tensor to (B,C,D,H,W) one-hot float.
    """
    B, _, D, H, W = tensor.shape
    oh = torch.zeros(B, num_classes, D, H, W,
                     dtype=torch.float32, device=tensor.device)
    oh.scatter_(1, tensor, 1.0)
    return oh


def _hd95_single(pred_bin: np.ndarray, gt_bin: np.ndarray,
                 spacing_mm: float = 1.0) -> float:
    """
    95th-percentile Hausdorff distance between two binary 3D masks.
    Returns np.nan if either mask is empty (class absent in prediction or GT).
    spacing_mm: isotropic voxel spacing in mm (BraTS preprocessed = 1mm).
    """
    pred_pts = np.argwhere(pred_bin)
    gt_pts   = np.argwhere(gt_bin)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")

    # Vectorised distances (works for 3D, reasonable for 128^3 patches)
    from scipy.spatial import cKDTree
    tree_gt   = cKDTree(gt_pts   * spacing_mm)
    tree_pred = cKDTree(pred_pts * spacing_mm)

    d_pred_to_gt, _ = tree_gt.query(pred_pts * spacing_mm)
    d_gt_to_pred, _ = tree_pred.query(gt_pts  * spacing_mm)

    hd = max(np.percentile(d_pred_to_gt, 95),
             np.percentile(d_gt_to_pred, 95))
    return float(hd)


def compute_metrics(
    logits: torch.Tensor,    # (B, C, D, H, W)  raw model output
    labels: torch.Tensor,    # (B, 1, D, H, W)  int labels {0..4}
    num_classes: int = OUT_CHANNELS,
    compute_hd95: bool = True,
    spacing_mm: float = 1.0,
) -> dict:
    """
    Compute DSC, HD95, Sensitivity, Precision for each foreground class.

    Returns dict with keys:
        dsc        : list[float] length 4  — [NCR, ED, NET, ET]
        hd95       : list[float] length 4  — mm, nan if class absent
        sensitivity: list[float] length 4
        precision  : list[float] length 4
        dsc_mean   : float  (mean over non-nan classes)
    """
    B = logits.shape[0]
    probs = torch.softmax(logits, dim=1)                  # (B,C,D,H,W)
    preds = probs.argmax(dim=1, keepdim=True)             # (B,1,D,H,W)

    pred_oh = _onehot(preds,  num_classes)                # (B,C,D,H,W)
    gt_oh   = _onehot(labels, num_classes)                # (B,C,D,H,W)

    eps = 1e-8
    dsc_list, hd95_list, sens_list, prec_list = [], [], [], []

    for c in range(1, num_classes):   # skip background (class 0)
        p = pred_oh[:, c]   # (B,D,H,W)
        g = gt_oh[:, c]     # (B,D,H,W)

        tp = (p * g).sum().item()
        fp = (p * (1 - g)).sum().item()
        fn = ((1 - p) * g).sum().item()

        dsc  = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        sens = (tp + eps) / (tp + fn + eps)
        prec = (tp + eps) / (tp + fp + eps)

        dsc_list.append(dsc)
        sens_list.append(sens)
        prec_list.append(prec)

        # HD95 — computed per sample then averaged
        if compute_hd95:
            hd_batch = []
            pred_np = preds.cpu().numpy()   # (B,1,D,H,W)
            gt_np   = labels.cpu().numpy()
            for b in range(B):
                pred_bin = (pred_np[b, 0] == c)
                gt_bin   = (gt_np[b,  0] == c)
                hd_batch.append(_hd95_single(pred_bin, gt_bin, spacing_mm))
            valid = [v for v in hd_batch if not np.isnan(v)]
            hd95_list.append(float(np.mean(valid)) if valid else float("nan"))
        else:
            hd95_list.append(float("nan"))

    valid_dsc = [v for v in dsc_list if not np.isnan(v)]
    dsc_mean  = float(np.mean(valid_dsc)) if valid_dsc else float("nan")

    return {
        "dsc":         dsc_list,    # [NCR, ED, NET, ET]
        "hd95":        hd95_list,
        "sensitivity": sens_list,
        "precision":   prec_list,
        "dsc_mean":    dsc_mean,
    }


# ── Streaming metrics (avoids OOM from stacking all logits) ────────────────────
class StreamingMetrics:
    """Accumulates TP/FP/FN per-batch instead of stacking all logits on CPU."""
    def __init__(self, num_classes: int = OUT_CHANNELS):
        self.num_classes = num_classes
        self.eps = 1e-8
        self.tp_sum = [0.0] * (num_classes - 1)
        self.fp_sum = [0.0] * (num_classes - 1)
        self.fn_sum = [0.0] * (num_classes - 1)
        self.hd95_samples = [[] for _ in range(num_classes - 1)]

    def add_batch(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1, keepdim=True)
        B = logits.shape[0]
        pred_np = preds.cpu().numpy()
        gt_np = labels.cpu().numpy()
        for c_idx, c in enumerate(range(1, self.num_classes)):
            pred_bin = (preds == c).float()
            gt_bin = (labels == c).float()
            self.tp_sum[c_idx] += (pred_bin * gt_bin).sum().item()
            self.fp_sum[c_idx] += (pred_bin * (1 - gt_bin)).sum().item()
            self.fn_sum[c_idx] += ((1 - pred_bin) * gt_bin).sum().item()
            for b in range(B):
                self.hd95_samples[c_idx].append(
                    _hd95_single(pred_np[b, 0] == c, gt_np[b, 0] == c)
                )

    def compute(self):
        dsc_list, hd95_list, sens_list, prec_list = [], [], [], []
        for c_idx in range(self.num_classes - 1):
            tp, fp, fn = self.tp_sum[c_idx], self.fp_sum[c_idx], self.fn_sum[c_idx]
            dsc_list.append((2 * tp + self.eps) / (2 * tp + fp + fn + self.eps))
            sens_list.append((tp + self.eps) / (tp + fn + self.eps))
            prec_list.append((tp + self.eps) / (tp + fp + self.eps))
            valid_hd = [v for v in self.hd95_samples[c_idx] if not np.isnan(v)]
            hd95_list.append(float(np.mean(valid_hd)) if valid_hd else float('nan'))
        valid_dsc = [v for v in dsc_list if not np.isnan(v)]
        dsc_mean = float(np.mean(valid_dsc)) if valid_dsc else float('nan')
        return {
            'dsc': dsc_list, 'hd95': hd95_list,
            'sensitivity': sens_list, 'precision': prec_list, 'dsc_mean': dsc_mean,
        }


# ── FLOPs ─────────────────────────────────────────────────────────────────────
def count_flops(model: torch.nn.Module,
                input_shape: tuple = (1, 4, 128, 128, 128),
                device: torch.device = torch.device("cpu"),
                logger: logging.Logger = None) -> float:
    """
    Count GFLOPs for one forward pass.
    Catches ALL exceptions so it never crashes regardless of model state
    (e.g. torch.compile breaks JIT tracing used by fvcore).
    Returns -1.0 if counting fails.
    """
    dummy = torch.zeros(input_shape, device=device)

    # fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            gflops = flops.total() / 1e9
        if logger:
            logger.info(f"FLOPs (fvcore) : {gflops:.2f} GFLOPs")
        return gflops
    except Exception:
        pass

    # thop
    try:
        from thop import profile
        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = macs * 2 / 1e9
        if logger:
            logger.info(f"FLOPs (thop)   : {gflops:.2f} GFLOPs")
        return gflops
    except Exception:
        pass

    if logger:
        logger.warning(
            "FLOPs: skipped — fvcore/thop unavailable or incompatible"
            " (compiled model?). Set to -1.0."
        )
    return -1.0


# ── Epoch log helper ──────────────────────────────────────────────────────────
def log_epoch(logger, epoch, epochs, train_loss, val_loss, m, epoch_time, elapsed, lr):
    """Print one standardised epoch summary line."""
    eta_sec  = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
    eta_h    = int(eta_sec // 3600)
    eta_m    = int((eta_sec % 3600) // 60)

    logger.info(
        f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Mean DSC: {m['dsc_mean']:.4f}"
    )
    dsc  = m["dsc"]
    hd   = m["hd95"]
    sens = m["sensitivity"]
    prec = m["precision"]
    logger.info(
        f"  DSC   NCR/ED/NET/ET : "
        f"{dsc[0]:.4f} / {dsc[1]:.4f} / {dsc[2]:.4f} / {dsc[3]:.4f}"
    )
    logger.info(
        f"  HD95  NCR/ED/NET/ET : "
        f"{hd[0]:.2f} / {hd[1]:.2f} / {hd[2]:.2f} / {hd[3]:.2f}  mm"
    )
    logger.info(
        f"  Sens  NCR/ED/NET/ET : "
        f"{sens[0]:.4f} / {sens[1]:.4f} / {sens[2]:.4f} / {sens[3]:.4f}"
    )
    logger.info(
        f"  Prec  NCR/ED/NET/ET : "
        f"{prec[0]:.4f} / {prec[1]:.4f} / {prec[2]:.4f} / {prec[3]:.4f}"
    )
    logger.info(
        f"  Time: {epoch_time:.1f}s | Elapsed: {elapsed/3600:.2f}h | "
        f"ETA: {eta_h}h {eta_m}m | LR: {lr:.2e}"
    )


def make_epoch_record(epoch, train_loss, val_loss, m, lr, epoch_time, elapsed):
    """Return serialisable dict for results.json."""
    return {
        "epoch":         epoch,
        "train_loss":    train_loss,
        "val_loss":      val_loss,
        "dsc_mean":      m["dsc_mean"],
        "dsc":           m["dsc"],         # [NCR,ED,NET,ET]
        "hd95":          m["hd95"],
        "sensitivity":   m["sensitivity"],
        "precision":     m["precision"],
        "lr":            lr,
        "epoch_time_s":  epoch_time,
        "elapsed_hours": elapsed / 3600,
    }
