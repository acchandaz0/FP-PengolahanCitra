"""
evaluate_wt_tc_et.py — Post-hoc WT/TC/ET evaluation from a trained checkpoint.

Universal: works for ALL FIVE variants
    Baseline1 (unet), Baseline2 (unet), Ablasi1 (sk),
    Ablasi2 (mmsk),   Proposed  (mmsk).

Aggregation
-----------
For each region this script now reports BOTH:
    * MACRO  — per-case metric computed first, then averaged across cases
               (the BraTS-challenge convention). Reported as
               dsc_macro_mean / dsc_macro_std, etc.
    * MICRO  — global TP/FP/FN pooled across ALL cases, then the metric
               computed once. Reported as dsc_micro, sensitivity_micro,
               precision_micro.
HD95 has no micro form (it is a per-case distance metric), so only its
macro mean/std is reported.

The per-case arrays (per_case) are still stored in full for the later
Wilcoxon signed-rank tests.

Purpose
-------
All five ablation training scripts only log metrics on the RAW 5-class label
space (BG, NCR, ED, NET, ET) via StreamingMetrics. The thesis (Bab III,
Pasca-pemrosesan dan Evaluasi) commits to reporting BraTS-standard composite
regions WT/TC/ET instead. This script bridges that gap.

Usage
-----
    # Ablasi1 (SK / BGSK, bare state_dict checkpoint)
    python evaluate_wt_tc_et.py \
        --checkpoint results/ablation1_sk_tversky/best_model.pth \
        --dataset_json dataset.json \
        --architecture sk \
        --variant_name "Ablasi1" \
        --output_dir results/wt_tc_et_eval

    # Ablasi2 (MMSK + Dice)
    python evaluate_wt_tc_et.py \
        --checkpoint results/ablation2_mmsk_dice/best_model.pth \
        --dataset_json dataset.json \
        --architecture mmsk \
        --variant_name "Ablasi2" \
        --output_dir results/wt_tc_et_eval

    # Proposed (MMSK + Tversky), with gate interpretability
    python evaluate_wt_tc_et.py \
        --checkpoint results/proposed_mmsk_tversky/best_model.pth \
        --dataset_json dataset.json \
        --architecture mmsk \
        --variant_name "Proposed" \
        --output_dir results/wt_tc_et_eval \
        --log-gate

    # Baseline1 / Baseline2 (MONAI 3D U-Net)
    #   --architecture unet   (omit --log-gate)

Output
------
    <output_dir>/<variant_name>_wt_tc_et.json
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
               out_channels: int = 5, store_attention: bool = False,
               logger=None):
    """
    Instantiate the correct architecture class and load weights.

    store_attention only matters for MMSK: set True ONLY when --log-gate is on,
    so that the forward pass returns (logits, attn_maps). When False, MMSK
    returns a plain logits tensor (identical to how the training scripts run
    validate()), which avoids the tuple-unpack bug.
    """
    if architecture == "mmsk":
        from mmsk_3d_unet import MMSK3DUNet
        model = MMSK3DUNet(in_channels=4, out_channels=out_channels,
                           store_attention=store_attention)

    elif architecture == "unet":
        # MONAI standard 3D U-Net, matches Baseline 1 / Baseline 2 config
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3, in_channels=4, out_channels=out_channels,
            channels=(32, 64, 128, 256), strides=(2, 2, 2),
            num_res_units=2,
        )

    elif architecture == "sk":
        # SK-3D U-Net (BGSK, no modal gate) — matches Ablasi1 training script
        from bgsk_3d_unet_patched import BGSK3DUNet
        model = BGSK3DUNet(in_channels=4, out_channels=out_channels)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Two checkpoint formats:
    #   - full dict  (Baseline1, Ablasi2, Proposed): has "model_state_dict"
    #   - bare state_dict (Ablasi1): OrderedDict of tensors, no wrapper keys
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", None)
    else:
        state_dict = ckpt              # bare state_dict
        epoch = None

    # Strip a possible "module." prefix (DataParallel), harmless if absent
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # Informative diagnosis instead of a raw traceback — helps spot the
        # known checkpoint/architecture mismatches (e.g. sk_bn present/absent).
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        msg = (
            f"load_state_dict FAILED for architecture='{architecture}'.\n"
            f"  model has {len(model_keys)} keys, checkpoint has {len(ckpt_keys)} keys.\n"
            f"  missing in checkpoint ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected in checkpoint ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}\n"
            f"  -> the model class does not match this checkpoint's architecture.\n"
            f"     Check the correct model module/class, or the sk_bn variant."
        )
        if logger is not None:
            logger.error(msg)
        raise RuntimeError(msg) from e

    model.to(device).eval()

    epoch_1indexed = (epoch + 1) if epoch is not None else None
    return model, epoch_1indexed


# ── Region conversion (mirrors alg:wt_tc_et_conversion) ────────────────────────
def to_region_masks(label_volume: np.ndarray) -> dict:
    """label_volume: (D,H,W) int {0..4} -> dict region -> binary (D,H,W) bool."""
    return {region: np.isin(label_volume, raw_labels)
            for region, raw_labels in REGION_RAW_LABELS.items()}


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
    """
    DSC, HD95, Sensitivity, Precision for one region, one case.
    Also returns the raw TP/FP/FN voxel counts so the caller can pool them
    across cases for the MICRO aggregation.
    """
    eps = 1e-8
    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    fp = float(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
    fn = float(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())

    dsc = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    sens = (tp + eps) / (tp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    hd95 = hd95_single(pred_bin, gt_bin, spacing_mm)

    return {"dsc": dsc, "hd95": hd95, "sensitivity": sens, "precision": prec,
            "tp": tp, "fp": fp, "fn": fn}


# ── Main evaluation loop ─────────────────────────────────────────────────────────
def evaluate(model, loader, device, architecture: str, log_gate: bool,
             logger) -> dict:
    per_case = {r: {"dsc": [], "hd95": [], "sensitivity": [], "precision": []}
                for r in REGIONS}

    # MICRO accumulators: global TP/FP/FN pooled across every case.
    micro_acc = {r: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for r in REGIONS}
    # GT presence bookkeeping (how many cases actually contain the region).
    gt_present = {r: 0 for r in REGIONS}
    gt_empty = {r: 0 for r in REGIONS}

    gate_layer_key = "bottleneck"
    gate_sub_idx = 1  # 0 = mmsk1, 1 = mmsk2 (second conv in the block)
    gate_acc = {r: {"a1_sum": 0.0, "a2_sum": 0.0, "n_voxels": 0} for r in REGIONS}

    n_cases = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)   # (1,4,D,H,W)
            labels = batch["label"].to(device)   # (1,1,D,H,W)

            # ── Robust forward: works whether the model returns a plain
            #    tensor or a (logits, attn_maps) tuple/list. Fixes the
            #    tuple-unpack bug for MMSK. ────────────────────────────────
            out = model(images)
            if isinstance(out, (tuple, list)):
                logits = out[0]
                attn_maps = out[1] if len(out) > 1 else None
            else:
                logits = out
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
                # Pool raw counts for micro aggregation
                micro_acc[region]["tp"] += m["tp"]
                micro_acc[region]["fp"] += m["fp"]
                micro_acc[region]["fn"] += m["fn"]
                # Track GT presence for this region
                if gt_masks[region].sum() > 0:
                    gt_present[region] += 1
                else:
                    gt_empty[region] += 1

            # ── Gate weight logging (Select attention [a1,a2], MMSK only) ──
            if architecture == "mmsk" and log_gate and attn_maps is not None:
                try:
                    attn_pair = attn_maps[gate_layer_key][gate_sub_idx]  # [1,2,C]
                    a1 = attn_pair[0, 0].mean().item()
                    a2 = attn_pair[0, 1].mean().item()
                    for region in REGIONS:
                        voxel_count = int(gt_masks[region].sum())
                        if voxel_count > 0:
                            gate_acc[region]["a1_sum"] += a1 * voxel_count
                            gate_acc[region]["a2_sum"] += a2 * voxel_count
                            gate_acc[region]["n_voxels"] += voxel_count
                except (KeyError, IndexError, TypeError) as e:
                    if batch_idx == 0:
                        logger.warning(
                            f"Gate logging skipped: attn_maps structure "
                            f"unexpected ({e}). Metrics unaffected; adjust "
                            f"gate_layer_key/gate_sub_idx to match MMSK3DUNet."
                        )

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

    return per_case, micro_acc, gt_present, gt_empty, gate_stats, n_cases


def summarize(per_case: dict, micro_acc: dict, gt_present: dict,
              gt_empty: dict, n_cases: int) -> dict:
    """
    Build the per-region summary with BOTH aggregations:
        MACRO  — mean/std over per-case values (NaN ignored, for HD95).
        MICRO  — computed from globally pooled TP/FP/FN.
    """
    eps = 1e-8
    summary = {}
    for region in REGIONS:
        summary[region] = {}

        # ── MACRO (per-case mean/std) ──────────────────────────────────────
        for key in ("dsc", "hd95", "sensitivity", "precision"):
            vals = np.array(per_case[region][key], dtype=float)
            valid = vals[~np.isnan(vals)]
            summary[region][f"{key}_macro_mean"] = (
                float(np.mean(valid)) if len(valid) else float("nan"))
            summary[region][f"{key}_macro_std"] = (
                float(np.std(valid)) if len(valid) else float("nan"))
            summary[region][f"{key}_n_valid"] = int(len(valid))

        # ── MICRO (global pooled TP/FP/FN) ─────────────────────────────────
        tp = micro_acc[region]["tp"]
        fp = micro_acc[region]["fp"]
        fn = micro_acc[region]["fn"]
        summary[region]["dsc_micro"] = float(
            (2 * tp + eps) / (2 * tp + fp + fn + eps))
        summary[region]["sensitivity_micro"] = float((tp + eps) / (tp + fn + eps))
        summary[region]["precision_micro"] = float((tp + eps) / (tp + fp + eps))

        # ── Case bookkeeping ───────────────────────────────────────────────
        summary[region]["n_cases_total"] = int(n_cases)
        summary[region]["n_cases_gt_present"] = int(gt_present[region])
        summary[region]["n_cases_gt_empty"] = int(gt_empty[region])

    return summary


# ── Entry point ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc WT/TC/ET evaluation from a trained checkpoint "
                    "(universal: unet / sk / mmsk; reports micro + macro)."
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
    parser.add_argument("--num_workers", type=int, default=4)
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
    model, epoch = load_model(
        args.architecture, checkpoint_path, device,
        store_attention=args.log_gate, logger=logger,
    )
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
                            num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Validation cases: {len(val_loader)}")

    # ── Evaluate ────────────────────────────────────────────────────────────
    logger.info("Starting WT/TC/ET evaluation ...")
    per_case, micro_acc, gt_present, gt_empty, gate_stats, n_cases = evaluate(
        model, val_loader, device, args.architecture, args.log_gate, logger
    )
    summary = summarize(per_case, micro_acc, gt_present, gt_empty, n_cases)

    for region in REGIONS:
        s = summary[region]
        logger.info(
            f"  {region}: "
            f"DSC[macro={s['dsc_macro_mean']:.4f} / micro={s['dsc_micro']:.4f}]  "
            f"HD95={s['hd95_macro_mean']:.2f}mm  "
            f"Sens[macro={s['sensitivity_macro_mean']:.4f} / micro={s['sensitivity_micro']:.4f}]  "
            f"Prec[macro={s['precision_macro_mean']:.4f} / micro={s['precision_micro']:.4f}]  "
            f"(gt_present={s['n_cases_gt_present']}/{s['n_cases_total']}, "
            f"gt_empty={s['n_cases_gt_empty']})"
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
        "architecture": args.architecture,
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