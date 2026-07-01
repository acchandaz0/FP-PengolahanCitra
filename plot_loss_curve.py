"""
plot_loss_curve.py — Plot training-loss & validation-DSC curve from a
results.json (or a recovered {"results": [...]} file) produced by any of the
five ablation training scripts.

Usage:
    python plot_loss_curve.py \
        --results results/ablation1_sk_tversky/results_recovered.json \
        --title   "Ablasi 1 (SK 3D U-Net + Tversky)" \
        --out      gambar/loss_ablasi1.png

Works for both formats:
    * full results.json  (has "results" key among many others)
    * recovered file      ({"results": [...]})
Each element of results[] must contain: epoch, train_loss, val_loss, dsc_mean.
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to results.json / recovered json")
    ap.add_argument("--title", default="", help="figure title")
    ap.add_argument("--out", required=True, help="output image path (.png/.pdf)")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    R = data["results"] if "results" in data else data
    if not R:
        raise SystemExit("results kosong — tidak ada yang bisa diplot.")

    epochs     = [r["epoch"] + 1 for r in R]          # 1-indexed for display
    train_loss = [r["train_loss"] for r in R]
    val_loss   = [r["val_loss"]   for r in R]
    val_dsc    = [r["dsc_mean"]   for r in R]

    best_i   = max(range(len(R)), key=lambda i: val_dsc[i])
    best_ep  = epochs[best_i]
    best_dsc = val_dsc[best_i]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))

    # ── Loss (left axis) ──
    ax1.plot(epochs, train_loss, color="#1f77b4", lw=1.4, label="Train Loss")
    ax1.plot(epochs, val_loss,   color="#1f77b4", lw=1.4, ls="--", alpha=0.7,
             label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.25)

    # ── Val DSC (right axis) ──
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_dsc, color="#d62728", lw=1.6, label="Val Mean DSC")
    ax2.set_ylabel("Validation Mean DSC", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 1)

    # ── mark best epoch ──
    ax2.axvline(best_ep, color="gray", ls=":", lw=1)
    ax2.scatter([best_ep], [best_dsc], color="#d62728", zorder=5, s=30)
    ax2.annotate(f"best: ep {best_ep}\nDSC {best_dsc:.4f}",
                 xy=(best_ep, best_dsc),
                 xytext=(best_ep, min(best_dsc + 0.12, 0.95)),
                 ha="center", fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # ── combined legend ──
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8, framealpha=0.9)

    if args.title:
        plt.title(args.title, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved -> {args.out}  (best epoch {best_ep}, DSC {best_dsc:.4f}, "
          f"{len(R)} epochs)")


if __name__ == "__main__":
    main()