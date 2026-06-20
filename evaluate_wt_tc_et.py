"""
evaluate_wt_tc_et.py — Post-hoc WT/TC/ET evaluation from a trained checkpoint.

Purpose
-------
All five ablation training scripts (baseline1/2, ablation1/2, proposed)
only log metrics on the RAW 5-class label space (BG, NCR, ED, NET, ET),
via StreamingMetrics in brats_utils.py. The thesis (Bab III, Subbagian
Pasca-pemrosesan dan Evaluasi) commits to reporting BraTS-standard
composite regions WT/TC/ET instead. This script bridges that gap by:

    1. Loading a trained checkpoint (best_model.pth or any .pth)
    2. Re-running inference on the validation set
    3. Converting raw 5-class predictions -> WT/TC/ET binary maps
       (exactly mirrors Algoritma 3.x / alg:wt_tc_et_conversion in Bab III)
    4. Computing DSC, HD95, Sensitivity, Precision PER CASE PER REGION
       (per-case scores are required later for Wilcoxon signed-rank tests
       across variants — alg:wt_tc_et_metrics keeps only the mean, this
       script keeps the full array)
    5. (Optional, --log-gate) Logging mean [a1, a2] Select weights per
       region, for the interpretability table (only meaningful for MMSK
       variants: Ablation 2 / Proposed; pass --architecture mmsk)

Usage
-----
    python evaluate_wt_tc_et.py \
        --checkpoint /home/mci/arsyadl/mmsk/output_proposed/best_model.pth \
        --dataset_json /home/mci/arsyadl/mmsk/dataset_split.json \
        --architecture mmsk \
        --variant_name "Proposed" \
        --output_dir /home/mci/arsyadl/mmsk/wt_tc_et_results \
        --log-gate

    # For non-gated variants (Baseline 1/2), omit --log-gate; architecture
    # can be "unet" (MONAI 3D U-Net) or "sk" (SK-3D U-Net, no MMSK gate).

Output
------
    <output_dir>/<variant_name>_wt_tc_et.json
        {
          "variant": "Proposed",
          "checkpoint_epoch": 39,
          "n_cases": 269,
          "summary": {
              "WT": {"dsc_mean":..., "dsc_std":..., "hd95_mean":...,
                      "sensitivity_mean":..., "precision_mean":...},
              "TC": {...}, "ET": {...}
          },
          "per_case": {
              "WT": {"dsc": [...269 floats...], "hd95": [...], ...},
              "TC": {...}, "ET": {...}
          },
          "gate_weights": {                      # only if --log-gate
              "WT": {"a1_mean":..., "a2_mean":...},
              "TC": {...}, "ET": {...}
          } or null
        }

This per-case JSON is the direct input for a later aggregation script
that builds tab:perbandingan_5varian and runs scipy.stats.wilcoxon for
tab:wilcoxon.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree

from brats_utils import BraTSNpzDataset, validate_npz_files, setup_logger


# ── Composite region definitions (mirrors alg:wt_tc_et_conversion) ─────────────
# Raw label convention: 0=BG, 1=NCR, 2=ED, 3=NET, 4=ET
REGION_RAW_LABELS = {
    "WT": (1, 2, 3, 4),   # NCR + ED + NET + ET
    "TC": (1, 3, 4),      # NCR + NET + ET
    "ET": (4,),           # ET only
}
REGIONS = ["WT", "TC", "ET"]


# ── Model loading (architecture-agnostic) ───────────────────────────────────────
def load_model(architecture: str, checkpoint_path: Path, device: torch.device,
                out_channels: int = 5):
    """
    Instantiate the correct architecture class and load weights.
    Extend this if variant scripts use different constructor signatures.
    """
    if architecture == "mmsk":
        from mmsk_3d_unet import MMSK3DUNet
        model = MMSK3DUNet(in_channels=4, out_channels=out_channels,
                            store_attention=True)
    elif architecture == "unet":
        # MONAI standard 3D U-Net, matches Baseline 1 / Baseline 2 config
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3, in_channels=4, out_channels=out_channels,
            channels=(32, 64, 128, 256), strides=(2, 2, 2),
            num_res_units=2,
        )
    elif architecture == "sk":
        # TODO: import your SK-3D U-Net / BGSK class for Ablation 1 here.
        # Left unimplemented since the class name/module was not provided.
        raise NotImplementedError(
            "architecture='sk' needs the actual Ablation 1 model class — "
            "import it here the same way 'mmsk' imports MMSK3DUNet."
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()

    epoch = ckpt.get("epoch", None)
    epoch_1indexed = (epoch + 1) if epoch is not None else None
    return model, epoch_1indexed


# ── Region conversion (mirrors alg:wt_tc_et_conversion) ────────────────────────
def to_region_masks(label_volume: np.ndarray) -> dict:
    """
    label_volume: (D,H,W) int array with raw values {0,1,2,3,4}
    Returns dict region_name -> binary (D,H,W) bool array.
    """
    masks = {}
    for region, raw_labels in REGION_RAW_LABELS.items():
        masks[region] = np.isin(label_volume, raw_labels)
    return masks


# ── HD95 (same implementation as brats_utils._hd95_single) ─────────────────────
def hd95_single(pred_bin: np.ndarray, gt_bin: np.ndarray,
                 spacing_mm: float = 1.0) -> float:
    pred_pts = np.argwhere(pred_bin)
    gt_pts = np.argwhere(gt_bin)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")
    tree_gt = cKDTree(gt_pts * spacing_mm)
    tree_pred = cKDTree(pred_pts * spacing_mm)
    d_pred_to_gt, _ = tree_gt.query(pred_pts * spacing_mm)
    d_gt_to_pred, _ = tree_pred.query(gt_pts * spacing_mm)
    return float(max(np.percentile(d_pred_to_gt, 95),
                      np.percentile(d_gt_to_pred, 95)))


def region_case_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray,
                         spacing_mm: float = 1.0) -> dict:
    """DSC, HD95, Sensitivity, Precision for one region, one case."""
    eps = 1e-8
    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    fp = float(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
    fn = float(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())

    dsc = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    sens = (tp + eps) / (tp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    hd95 = hd95_single(pred_bin, gt_bin, spacing_mm)

    return {"dsc": dsc, "hd95": hd95, "sensitivity": sens, "precision": prec}


# ── Main evaluation loop ─────────────────────────────────────────────────────────
def evaluate(model, loader, device, architecture: str, log_gate: bool,
             logger) -> dict:
    """
    Returns:
        per_case   : {region: {metric: [n_cases floats]}}
        gate_stats : {region: {"a1_mean":..., "a2_mean":..., "n_voxels":...}} or None
    """
    per_case = {r: {"dsc": [], "hd95": [], "sensitivity": [], "precision": []}
                for r in REGIONS}

    # Gate weight accumulation: sum of (a1,a2) over voxels belonging to each
    # region, divided by voxel count at the end -> spatial mean per region.
    # Uses the LAST MMSK conv layer in the bottleneck as the representative
    # attention map (deepest semantic level); adjust `gate_layer_key` /
    # `gate_sub_idx` below if a different layer is preferred.
    gate_layer_key = "bottleneck"
    gate_sub_idx = 1  # 0 = mmsk1, 1 = mmsk2 (second conv in the block)
    gate_acc = {r: {"a1_sum": 0.0, "a2_sum": 0.0, "n_voxels": 0} for r in REGIONS}

    n_cases = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)   # (1,4,D,H,W)
            labels = batch["label"].to(device)    # (1,1,D,H,W)

            if architecture == "mmsk" and log_gate:
                logits, attn_maps = model(images)
            else:
                logits = model(images)
                attn_maps = None

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).squeeze(0).cpu().numpy()   # (D,H,W)
            gt = labels.squeeze(0).squeeze(0).cpu().numpy()        # (D,H,W)

            pred_masks = to_region_masks(preds)
            gt_masks = to_region_masks(gt)

            for region in REGIONS:
                m = region_case_metrics(pred_masks[region], gt_masks[region])
                for key in ("dsc", "hd95", "sensitivity", "precision"):
                    per_case[region][key].append(m[key])

            # ── Gate weight logging (Select attention [a1,a2], MMSK only) ──
            if architecture == "mmsk" and log_gate and attn_maps is not None:
                # attn_maps[gate_layer_key] is a tuple (attn1, attn2), each
                # shaped [B, M=2, out_ch]. We take the mean over out_ch to
                # get a single (a1,a2) pair per voxel-region — but since SK
                # attention here is channel-wise (not spatial), the same
                # [a1,a2] pair applies to the whole volume for this sample.
                # We therefore weight by region voxel COUNT (not per-voxel
                # attention, since none exists at this granularity) to get
                # a region-conditional average across the dataset.
                attn_pair = attn_maps[gate_layer_key][gate_sub_idx]  # [1,2,C]
                a1 = attn_pair[0, 0].mean().item()
                a2 = attn_pair[0, 1].mean().item()
                for region in REGIONS:
                    voxel_count = int(gt_masks[region].sum())
                    if voxel_count > 0:
                        gate_acc[region]["a1_sum"] += a1 * voxel_count
                        gate_acc[region]["a2_sum"] += a2 * voxel_count
                        gate_acc[region]["n_voxels"] += voxel_count

            n_cases += 1
            if (batch_idx + 1) % 25 == 0:
                logger.info(f"  Evaluated {batch_idx + 1}/{len(loader)} cases")

    gate_stats = None
    if architecture == "mmsk" and log_gate:
        gate_stats = {}
        for region in REGIONS:
            nv = gate_acc[region]["n_voxels"]
            if nv > 0:
                gate_stats[region] = {
                    "a1_mean": gate_acc[region]["a1_sum"] / nv,
                    "a2_mean": gate_acc[region]["a2_sum"] / nv,
                    "n_voxels": nv,
                }
            else:
                gate_stats[region] = {"a1_mean": float("nan"),
                                       "a2_mean": float("nan"), "n_voxels": 0}

    return per_case, gate_stats, n_cases


def summarize(per_case: dict) -> dict:
    """Compute mean/std per region/metric, ignoring NaN (HD95 on absent regions)."""
    summary = {}
    for region in REGIONS:
        summary[region] = {}
        for key in ("dsc", "hd95", "sensitivity", "precision"):
            vals = np.array(per_case[region][key], dtype=float)
            valid = vals[~np.isnan(vals)]
            summary[region][f"{key}_mean"] = float(np.mean(valid)) if len(valid) else float("nan")
            summary[region][f"{key}_std"] = float(np.std(valid)) if len(valid) else float("nan")
            summary[region][f"{key}_n_valid"] = int(len(valid))
    return summary


# ── Entry point ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc WT/TC/ET evaluation from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_json", type=str, required=True,
                         help="Same JSON used for training, must have a 'val' key.")
    parser.add_argument("--architecture", type=str, required=True,
                         choices=["mmsk", "unet", "sk"])
    parser.add_argument("--variant_name", type=str, required=True,
                         help='e.g. "Baseline1", "Baseline2", "Ablasi1", '
                              '"Ablasi2", "Proposed"')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log-gate", action="store_true",
                         help="Log mean [a1,a2] per region. MMSK architecture only.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"eval_{args.variant_name}", output_dir)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Variant: {args.variant_name}  |  Architecture: {args.architecture}")

    if args.log_gate and args.architecture != "mmsk":
        logger.warning(
            "--log-gate requested but architecture != 'mmsk'; "
            "no Select attention exists for this architecture. Disabling."
        )
        args.log_gate = False

    # ── Load model ──────────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    model, epoch = load_model(args.architecture, checkpoint_path, device)
    logger.info(f"Loaded checkpoint from epoch {epoch}")

    # ── Load validation data ───────────────────────────────────────────────
    with open(args.dataset_json) as f:
        dataset = json.load(f)
    val_files = validate_npz_files(dataset["val"], logger)
    if not val_files:
        logger.error("No valid validation files after validation scan.")
        sys.exit(1)
    val_ds = BraTSNpzDataset(val_files, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True)
    logger.info(f"Validation cases: {len(val_loader)}")

    # ── Evaluate ────────────────────────────────────────────────────────────
    logger.info("Starting WT/TC/ET evaluation ...")
    per_case, gate_stats, n_cases = evaluate(
        model, val_loader, device, args.architecture, args.log_gate, logger
    )
    summary = summarize(per_case)

    for region in REGIONS:
        logger.info(
            f"  {region}: DSC={summary[region]['dsc_mean']:.4f} "
            f"HD95={summary[region]['hd95_mean']:.2f}mm "
            f"Sens={summary[region]['sensitivity_mean']:.4f} "
            f"Prec={summary[region]['precision_mean']:.4f} "
            f"(n_valid={summary[region]['dsc_n_valid']}/{n_cases})"
        )
    if gate_stats is not None:
        for region in REGIONS:
            logger.info(
                f"  Gate {region}: a1={gate_stats[region]['a1_mean']:.4f} "
                f"a2={gate_stats[region]['a2_mean']:.4f}"
            )

    # ── Save output ─────────────────────────────────────────────────────────
    out = {
        "variant": args.variant_name,
        "checkpoint_epoch": epoch,
        "n_cases": n_cases,
        "summary": summary,
        "per_case": per_case,
        "gate_weights": gate_stats,
    }
    out_path = output_dir / f"{args.variant_name}_wt_tc_et.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved results -> {out_path}")


if __name__ == "__main__":
    main()
