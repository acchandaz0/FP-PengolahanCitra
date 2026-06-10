"""
MMSK Gate Weight Visualization — Interpretability Analysis
Supports Research Question RQ5:
    "Can MMSK per-modality attention weights reveal which MRI modalities
     drive kernel selection for each tumor sub-region?"

Expected findings (Hypothesis H3):
    ET (Enhancing Tumor)   → T1ce-dominated gate weights
    WT (Whole Tumor)       → FLAIR-dominated gate weights
    TC (Tumor Core)        → T1 + T1ce combined

Usage:
    python visualize_mmsk_attention.py \
        --checkpoint ./output_proposed_mmsk_tversky/best_model.pth \
        --dataset_json ./dataset.json \
        --output_dir ./attention_maps \
        --num_cases 20 \
        --gpu 0
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import json
import argparse

from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, CropForegroundd, ScaleIntensityRanged,
)

from mmsk_3d_unet import MMSK3DUNet


MODALITY_NAMES = ["T1", "T2", "FLAIR", "T1ce"]
CLASS_NAMES    = ["NCR", "ED", "ET"]       # excluding background
REGION_NAMES   = {
    1: "WT (Whole Tumor)",
    2: "TC (Tumor Core)",
    3: "ET (Enhancing Tumor)",
}

LAYER_NAMES = ["enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1"]


# ─────────────────────────────────────────────────────────────────────────────
def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1, b_min=0.0, b_max=1.0, clip=True),
    ])


# ─────────────────────────────────────────────────────────────────────────────
def extract_gate_weights(model, image, device):
    """
    Run a single forward pass with store_attention=True and collect
    the SK attention weights (a1 vs a2 per branch) from every MMSK block.

    Returns:
        logits : [1, 4, D, H, W]
        attn   : dict layer_name → (attn_layer1, attn_layer2)
                 each attn_layer: [1, M=2, out_ch]
    """
    model.store_attention = True
    model.eval()
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)          # [1, 4, D, H, W]
        logits, attn = model(x)
    model.store_attention = False
    return logits, attn


# ─────────────────────────────────────────────────────────────────────────────
def compute_branch_preference(attn_maps):
    """
    For each layer, compute the mean branch preference across channels.

    attn_layer shape: [B=1, M=2, out_ch]
    Returns: dict layer_name → [branch0_mean, branch1_mean]
    """
    preferences = {}
    for layer_name in LAYER_NAMES:
        attn_pair = attn_maps[layer_name]          # tuple of (attn1, attn2)
        # Average the two MMSK convolutions in the block
        a1_mean = attn_pair[0][0].mean(dim=-1).cpu().numpy()   # [M]
        a2_mean = attn_pair[1][0].mean(dim=-1).cpu().numpy()   # [M]
        preferences[layer_name] = (a1_mean + a2_mean) / 2.0
    return preferences


# ─────────────────────────────────────────────────────────────────────────────
def compute_region_averaged_preferences(all_preferences, all_labels, all_logits, device):
    """
    Average gate weights separately for voxels belonging to ET, TC, WT.
    This is the key interpretability analysis for RQ5.

    Since gate weights are channel-averaged scalars (not spatial maps),
    we aggregate per sample and then average per region class.

    Returns:
        region_prefs: dict region_label → dict layer_name → [branch0, branch1]
    """
    region_prefs = {r: {l: [] for l in LAYER_NAMES} for r in [1, 2, 3]}

    for prefs, label, logit in zip(all_preferences, all_labels, all_logits):
        # Check which regions exist in this sample
        label_np = label.squeeze().cpu().numpy() if torch.is_tensor(label) else label
        for region_id in [1, 2, 3]:
            if np.any(label_np == region_id):
                for layer_name in LAYER_NAMES:
                    region_prefs[region_id][layer_name].append(prefs[layer_name])

    # Average across cases
    avg_region_prefs = {}
    for region_id in [1, 2, 3]:
        avg_region_prefs[region_id] = {}
        for layer_name in LAYER_NAMES:
            if region_prefs[region_id][layer_name]:
                avg_region_prefs[region_id][layer_name] = np.mean(
                    region_prefs[region_id][layer_name], axis=0
                )
            else:
                avg_region_prefs[region_id][layer_name] = np.array([0.5, 0.5])

    return avg_region_prefs


# ─────────────────────────────────────────────────────────────────────────────
def plot_branch_preference_by_layer(avg_prefs, output_dir):
    """
    Bar chart: Branch 0 (small 3×3×3) vs Branch 1 (large 5×5×5)
    per encoder/decoder layer, grouped by tumor region (WT, TC, ET).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(
        "MMSK Branch Preference per Layer\n"
        "Branch 0 = small (3×3×3)  |  Branch 1 = large (5×5×5 effective)",
        fontsize=13, fontweight='bold'
    )

    colors = {'Branch 0 (small)': '#2196F3', 'Branch 1 (large)': '#FF5722'}
    x = np.arange(len(LAYER_NAMES))
    width = 0.35

    for ax, (region_id, region_name) in zip(axes, REGION_NAMES.items()):
        prefs = avg_prefs[region_id]
        b0 = [prefs[l][0] for l in LAYER_NAMES]
        b1 = [prefs[l][1] for l in LAYER_NAMES]

        bars0 = ax.bar(x - width/2, b0, width, label='Branch 0 (small 3×3×3)', color='#2196F3', alpha=0.85)
        bars1 = ax.bar(x + width/2, b1, width, label='Branch 1 (large 5×5×5)', color='#FF5722', alpha=0.85)

        ax.set_title(f"{region_name}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Network Layer")
        ax.set_ylabel("Mean Attention Weight")
        ax.set_xticks(x)
        ax.set_xticklabels(LAYER_NAMES, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Equal preference')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "branch_preference_by_layer.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_branch_preference_by_region(avg_prefs, output_dir):
    """
    Radar / grouped bar: For each region, which branch dominates?
    Key plot for hypothesis H3 validation.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Mean Branch Preference by Tumor Sub-Region\n"
        "H3: ET→small branch, WT→large branch",
        fontsize=12, fontweight='bold'
    )

    region_ids   = list(REGION_NAMES.keys())
    region_labels = list(REGION_NAMES.values())
    x = np.arange(len(region_ids))
    width = 0.35

    # Average branch preference across all layers per region
    b0_means = []
    b1_means = []
    for region_id in region_ids:
        b0_vals = [avg_prefs[region_id][l][0] for l in LAYER_NAMES]
        b1_vals = [avg_prefs[region_id][l][1] for l in LAYER_NAMES]
        b0_means.append(np.mean(b0_vals))
        b1_means.append(np.mean(b1_vals))

    bars0 = ax.bar(x - width/2, b0_means, width, label='Branch 0 (small 3×3×3)', color='#2196F3', alpha=0.85)
    bars1 = ax.bar(x + width/2, b1_means, width, label='Branch 1 (large 5×5×5)', color='#FF5722', alpha=0.85)

    # Annotate values
    for bar in bars0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Tumor Sub-Region", fontsize=11)
    ax.set_ylabel("Mean Attention Weight (across all layers)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Equal preference')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "branch_preference_by_region.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmap_layer_vs_region(avg_prefs, output_dir):
    """
    Heatmap: rows = layers, columns = regions, value = Branch 0 weight
    (> 0.5 → small kernel preferred, < 0.5 → large kernel preferred)
    """
    data = np.zeros((len(LAYER_NAMES), 3))
    for j, region_id in enumerate([1, 2, 3]):
        for i, layer_name in enumerate(LAYER_NAMES):
            data[i, j] = avg_prefs[region_id][layer_name][0]  # branch 0 weight

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        "Small-Kernel Branch (3×3×3) Preference Heatmap\n"
        "Higher = small kernel preferred (fine detail mode)",
        fontsize=11, fontweight='bold'
    )

    im = ax.imshow(data, cmap='RdYlGn', vmin=0.3, vmax=0.7, aspect='auto')
    plt.colorbar(im, ax=ax, label='Branch 0 (small) attention weight')

    ax.set_xticks(range(3))
    ax.set_xticklabels([REGION_NAMES[r] for r in [1, 2, 3]], fontsize=9)
    ax.set_yticks(range(len(LAYER_NAMES)))
    ax.set_yticklabels(LAYER_NAMES, fontsize=9)
    ax.set_xlabel("Tumor Sub-Region")
    ax.set_ylabel("Network Layer")

    # Annotate cells
    for i in range(len(LAYER_NAMES)):
        for j in range(3):
            ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center',
                    fontsize=8, color='black' if 0.35 < data[i,j] < 0.65 else 'white')

    plt.tight_layout()
    save_path = output_dir / "branch_preference_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def save_summary_json(avg_prefs, output_dir):
    """Save numerical results for table generation in thesis."""
    summary = {}
    for region_id, region_name in REGION_NAMES.items():
        summary[region_name] = {}
        for layer_name in LAYER_NAMES:
            prefs = avg_prefs[region_id][layer_name]
            summary[region_name][layer_name] = {
                "branch0_small": float(prefs[0]),
                "branch1_large": float(prefs[1]),
                "dominant_branch": "small (3×3×3)" if prefs[0] > prefs[1] else "large (5×5×5)",
            }

    # Also compute per-region summary
    for region_id, region_name in REGION_NAMES.items():
        b0_all = [avg_prefs[region_id][l][0] for l in LAYER_NAMES]
        b1_all = [avg_prefs[region_id][l][1] for l in LAYER_NAMES]
        summary[region_name]["OVERALL"] = {
            "branch0_small_mean": float(np.mean(b0_all)),
            "branch1_large_mean": float(np.mean(b1_all)),
            "dominant_branch": "small (3×3×3)" if np.mean(b0_all) > np.mean(b1_all) else "large (5×5×5)",
        }

    save_path = output_dir / "gate_weight_summary.json"
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {save_path}")

    # Print human-readable table
    print("\n" + "="*65)
    print("MMSK Branch Preference Summary (H3 Validation)")
    print("="*65)
    print(f"{'Region':<25} {'Overall Branch 0 (small)':>24} {'Dominant':>12}")
    print("-"*65)
    for region_id, region_name in REGION_NAMES.items():
        ovr = summary[region_name]["OVERALL"]
        print(f"{region_name:<25} {ovr['branch0_small_mean']:>24.4f} {ovr['dominant_branch']:>12}")
    print("="*65)


# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    device     = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output dir : {output_dir}")

    # ── Load model ─────────────────────────────────────────────────────────
    model = MMSK3DUNet(in_channels=4, out_channels=5, store_attention=True).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    # Handle both raw state_dict and checkpoint dict
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({total_params:,} parameters)")

    # ── Load dataset (test split) ──────────────────────────────────────────
    with open(args.dataset_json) as f:
        dataset = json.load(f)

    test_files = dataset.get("test", dataset.get("val", []))
    test_files = test_files[:args.num_cases]
    print(f"Analyzing {len(test_files)} cases...")

    transforms = get_transforms()
    ds = Dataset(data=test_files, transform=transforms)

    # ── Collect attention maps per case ───────────────────────────────────
    all_preferences = []
    all_labels      = []
    all_logits      = []

    for i, sample in enumerate(ds):
        image = sample["image"]   # [4, D, H, W]
        label = sample["label"]   # [1, D, H, W]

        print(f"  Case {i+1}/{len(ds)} — image: {image.shape}, label: {label.shape}")

        logits, attn_maps = extract_gate_weights(model, image, device)
        prefs = compute_branch_preference(attn_maps)

        all_preferences.append(prefs)
        all_labels.append(label)
        all_logits.append(logits)

    # ── Aggregate by region ────────────────────────────────────────────────
    print("\nAggregating by tumor sub-region...")
    avg_prefs = compute_region_averaged_preferences(
        all_preferences, all_labels, all_logits, device
    )

    # ── Generate plots ─────────────────────────────────────────────────────
    print("\nGenerating visualizations...")
    plot_branch_preference_by_layer(avg_prefs, output_dir)
    plot_branch_preference_by_region(avg_prefs, output_dir)
    plot_heatmap_layer_vs_region(avg_prefs, output_dir)
    save_summary_json(avg_prefs, output_dir)

    print(f"\n✅ Attention analysis complete. Results saved to: {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MMSK Gate Weight Visualization for RQ5 Interpretability"
    )
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to trained MMSK model checkpoint (.pth)")
    parser.add_argument("--dataset_json", type=str, required=True,
                        help="Path to dataset JSON (must contain 'test' or 'val' split)")
    parser.add_argument("--output_dir",   type=str, default="./attention_maps")
    parser.add_argument("--num_cases",    type=int, default=20,
                        help="Number of test cases to analyze (default: 20, as per reliability plan)")
    parser.add_argument("--gpu",          type=int, default=0)
    args = parser.parse_args()

    main(args)
