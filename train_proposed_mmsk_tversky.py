"""
Proposed Model: MMSK-3D U-Net + Tversky Loss (α=0.3, β=0.7)

This is the FULL proposed method combining:
    1. MMSK Block — cross-modal gating using all 4 MRI modalities
       to adaptively select receptive field size per spatial location
    2. Tversky Loss (α=0.3, β=0.7) — asymmetric loss that penalizes
       false negatives more heavily (critical for ET ~1% volume)

Ablation Study Role:
    Proposed vs Ablation 1 (SK + Tversky)
    → Isolates the contribution of CROSS-MODAL GATING (MMSK vs plain SK)

    Proposed vs Ablation 2 (MMSK + Dice)
    → Isolates the contribution of TVERSKY LOSS on top of MMSK

    Proposed vs Baseline 1 (3D U-Net + Dice)
    → Full system gain (architecture + loss combined)

Expected outcome:
    H1: Significantly higher DSC_ET vs Baseline 1
    H2: Synergistic ET recall improvement > sum of individual parts
    H4: DSC_ET > 0.84, inference < 10s/volume

Metrics recorded per epoch:
    DSC        — Dice Similarity Coefficient per class (NCR/ED/NET/ET)
    HD95       — 95th-percentile Hausdorff Distance per class (mm)
    Sensitivity— Recall per class
    Precision  — Positive predictive value per class

Also recorded in results.json:
    FLOPs (GFLOPs), parameter count, inference time, training time
"""

import torch
import json
import time
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from monai.losses import TverskyLoss

from brats_utils import (
    BraTSNpzDataset,
    validate_npz_files,
    setup_logger,
    compute_metrics,
    count_flops,
    OUT_CHANNELS,
    FG_CLASS_NAMES,
)
from mmsk_3d_unet import MMSK3DUNet


class Trainer:
    def __init__(self, gpu_id: int, output_dir: str, vram_gb: float = 20.0):
        self.output_dir = Path(output_dir)
        self.logger     = setup_logger("proposed", self.output_dir)
        self.device     = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        # ── VRAM cap ──────────────────────────────────────────────────────
        if self.device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            fraction   = min(vram_gb / total_vram, 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, device=gpu_id)
            self.logger.info(
                f"VRAM cap       : {vram_gb:.0f} GB / {total_vram:.0f} GB "
                f"(fraction={fraction:.3f})"
            )

        self.logger.info(f"Device         : {self.device}")
        self.logger.info(f"Model          : MMSK-3D U-Net (cross-modal gating)")
        self.logger.info(f"Loss           : Tversky (α=0.3, β=0.7)   [PROPOSED]")
        self.logger.info(f"out_channels   : {OUT_CHANNELS}")
        self.logger.info(f"Label space    : BraTS 2024 raw {{0,1,2,3,4}}")
        self.logger.info(f"Output dir     : {self.output_dir}")

        # ── Full proposed architecture ─────────────────────────────────────
        self.model = MMSK3DUNet(
            in_channels     = 4,
            out_channels    = OUT_CHANNELS,
            store_attention = False,   # disable during training for speed
        ).to(self.device)

        # ── Tversky Loss ───────────────────────────────────────────────────
        # α=0.3 → lower FP penalty
        # β=0.7 → higher FN penalty (recall-focused, critical for tiny ET)
        self.loss_fn = TverskyLoss(
            to_onehot_y        = True,
            softmax            = True,
            alpha              = 0.3,
            beta               = 0.7,
            include_background = False,
        )

        # ── Optimizer / Scheduler ──────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=300,
        )

        # ── AMP ────────────────────────────────────────────────────────────
        self.use_amp     = self.device.type == "cuda"
        self.scaler      = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)
        self.accum_steps = 1   # set properly in run() after OOM probe

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"AMP             : {'ON' if self.use_amp else 'OFF (CPU)'}")

        # ── FLOPs (computed once at startup) ───────────────────────────────
        self.flops_giga = count_flops(
            self.model, device=self.device, logger=self.logger,
        )

        # ── Inference timing (populated after first validate call) ─────────
        self.inference_time_s = None

    # ──────────────────────────────────────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        epoch_loss = 0.0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss    = self.loss_fn(outputs, labels) / self.accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accum_steps == 0 or \
               (batch_idx + 1) == len(loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item() * self.accum_steps
            if batch_idx % 10 == 0:
                self.logger.debug(
                    f"  Batch {batch_idx:>4}/{len(loader)} | "
                    f"Loss: {loss.item() * self.accum_steps:.4f}"
                )

        return epoch_loss / len(loader)

    # ──────────────────────────────────────────────────────────────────────────
    def validate(self, loader: DataLoader):
        """
        Evaluate on validation set. Computes:
            - DSC, HD95, Sensitivity, Precision  (via compute_metrics)
            - Validation loss
            - Inference timing (on first call, warm-up then 5 timed passes)
        """
        self.model.eval()
        val_loss = 0.0

        # Accumulate all predictions and labels for metric computation
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    logits = self.model(inputs)
                    loss   = self.loss_fn(logits, labels)

                val_loss += loss.item()
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        # ── Compute comprehensive metrics on CPU ────────────────────────────
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        m = compute_metrics(logits_cat, labels_cat)

        # ── Inference timing (first epoch only) ────────────────────────────
        if self.inference_time_s is None:
            self.model.eval()
            dummy = torch.zeros(1, 4, 128, 128, 128, device=self.device)

            # Warm-up pass
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(dummy)

            # Timed passes
            torch.cuda.synchronize()
            n_timed  = 50
            t_start  = time.time()
            for _ in range(n_timed):
                with torch.no_grad():
                    _ = self.model(dummy)
            torch.cuda.synchronize()
            elapsed  = time.time() - t_start
            self.inference_time_s = elapsed / n_timed
            self.logger.info(
                f"Inference time: {self.inference_time_s*1000:.1f} ms/volume "
                f"(avg over {n_timed} passes)"
            )

        return val_loss / len(loader), m

    # ──────────────────────────────────────────────────────────────────────────
    def run(self, train_files: list[str], val_files: list[str], epochs: int = 300):
        self.logger.info("Pre-flight: validating .npz files ...")
        train_files = validate_npz_files(train_files, self.logger)
        val_files   = validate_npz_files(val_files,   self.logger)

        if not train_files:
            raise RuntimeError("No valid training files after validation scan.")
        if not val_files:
            raise RuntimeError("No valid validation files after validation scan.")

        train_ds = BraTSNpzDataset(train_files, augment=True)
        val_ds   = BraTSNpzDataset(val_files,   augment=False)

        _OOM = (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)

        def make_loaders(bs):
            tl = DataLoader(
                train_ds, batch_size=bs, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True,
            )
            vl = DataLoader(
                val_ds, batch_size=1, shuffle=False,
                num_workers=4, pin_memory=True,
            )
            return tl, vl

        # OOM probe: try bs=2+accum=2, fall back to bs=1+accum=4
        batch_size, self.accum_steps = 2, 2
        self.logger.info(
            f"Probing batch_size={batch_size}, accum_steps={self.accum_steps} ..."
        )
        train_loader, val_loader = make_loaders(batch_size)

        self.model.train()
        try:
            _b = next(iter(train_loader))
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                _out  = self.model(_b["image"].to(self.device))
                _loss = self.loss_fn(_out, _b["label"].to(self.device)) / self.accum_steps
            self.scaler.scale(_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            self.logger.info(
                f"Probe OK — bs={batch_size}, accum={self.accum_steps}, "
                f"effective_batch={batch_size * self.accum_steps}"
            )
        except _OOM:
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            batch_size, self.accum_steps = 1, 4
            train_loader, val_loader = make_loaders(batch_size)
            self.logger.warning(
                f"OOM at bs=2 — fallback: bs={batch_size}, "
                f"accum={self.accum_steps}, "
                f"effective_batch={batch_size * self.accum_steps}"
            )

        self.logger.info(f"Train batches  : {len(train_loader)} (bs={batch_size})")
        self.logger.info(f"Val batches    : {len(val_loader)}")
        self.logger.info(
            f"Accum steps    : {self.accum_steps} "
            f"(effective batch={batch_size * self.accum_steps})"
        )
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info("=" * 60)

        results   = []
        best_dsc  = 0.0
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs}")

            train_loss       = self.train_epoch(train_loader)
            val_loss, m      = self.validate(val_loader)
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            elapsed    = time.time() - start_time
            lr         = self.optimizer.param_groups[0]["lr"]

            # ── ETA estimate ──────────────────────────────────────────────
            eta_sec  = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            eta_h    = int(eta_sec // 3600)
            eta_m    = int((eta_sec % 3600) // 60)

            # ── Log epoch summary ─────────────────────────────────────────
            dsc  = m["dsc"]       # [NCR, ED, NET, ET]
            hd   = m["hd95"]
            sens = m["sensitivity"]
            prec = m["precision"]

            self.logger.info(
                f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Mean DSC: {m['dsc_mean']:.4f}"
            )
            self.logger.info(
                f"  DSC   NCR/ED/NET/ET : "
                f"{dsc[0]:.4f} / {dsc[1]:.4f} / {dsc[2]:.4f} / {dsc[3]:.4f}"
            )
            self.logger.info(
                f"  HD95  NCR/ED/NET/ET : "
                f"{hd[0]:.2f} / {hd[1]:.2f} / {hd[2]:.2f} / {hd[3]:.2f}  mm"
            )
            self.logger.info(
                f"  Sens  NCR/ED/NET/ET : "
                f"{sens[0]:.4f} / {sens[1]:.4f} / {sens[2]:.4f} / {sens[3]:.4f}"
            )
            self.logger.info(
                f"  Prec  NCR/ED/NET/ET : "
                f"{prec[0]:.4f} / {prec[1]:.4f} / {prec[2]:.4f} / {prec[3]:.4f}"
            )
            self.logger.info(
                f"  Time: {epoch_time:.1f}s | Elapsed: {elapsed/3600:.2f}h | "
                f"ETA: {eta_h}h {eta_m}m | LR: {lr:.2e}"
            )

            # ── Record ────────────────────────────────────────────────────
            results.append({
                "epoch":           epoch,
                "train_loss":      train_loss,
                "val_loss":        val_loss,
                "dsc_mean":        m["dsc_mean"],
                "dsc":             dsc,
                "hd95":            hd,
                "sensitivity":     sens,
                "precision":       prec,
                "lr":              lr,
                "epoch_time_s":    epoch_time,
                "elapsed_hours":   elapsed / 3600,
            })

            # ── Save best model ───────────────────────────────────────────
            if m["dsc_mean"] > best_dsc:
                best_dsc = m["dsc_mean"]
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "best_model.pth",
                )
                self.logger.info(f"  ✓ New best model saved (DSC: {best_dsc:.4f})")

            # ── Periodic checkpoint ───────────────────────────────────────
            if (epoch + 1) % 50 == 0:
                ckpt = self.output_dir / f"checkpoint_epoch{epoch+1}.pth"
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_dsc":             best_dsc,
                }, ckpt)
                self.logger.info(f"  Checkpoint saved → {ckpt.name}")

            self.logger.info("-" * 60)

        # ── Final summary ──────────────────────────────────────────────────
        total_time    = time.time() - start_time
        best_epoch_r  = max(results, key=lambda x: x["dsc_mean"])

        with open(self.output_dir / "results.json", "w") as f:
            json.dump({
                "model":                      "Proposed_MMSK_Tversky",
                "loss":                       "Tversky(alpha=0.3, beta=0.7)",
                "architecture":               "MMSK-3D U-Net (cross-modal gating)",
                "ablation_role":              "Full proposed method — MMSK + Tversky combined",
                "out_channels":               OUT_CHANNELS,
                "label_space":                "BraTS2024 raw {0,1,2,3,4}",
                "fg_class_names":             FG_CLASS_NAMES,
                "total_parameters":           sum(p.numel() for p in self.model.parameters()),
                "flops_giga":                 self.flops_giga,
                "inference_time_s":           self.inference_time_s,
                "batch_size":                 batch_size,
                "accum_steps":                self.accum_steps,
                "effective_batch":            batch_size * self.accum_steps,
                "best_dsc":                   best_dsc,
                "best_epoch":                 best_epoch_r["epoch"] + 1,
                "best_dsc_per_class":         best_epoch_r["dsc"],
                "best_hd95_per_class":        best_epoch_r["hd95"],
                "best_sensitivity_per_class": best_epoch_r["sensitivity"],
                "best_precision_per_class":   best_epoch_r["precision"],
                "total_time_hours":           total_time / 3600,
                "results":                    results,
            }, f, indent=2)

        self.logger.info("=" * 60)
        self.logger.info("Training completed! [Proposed: MMSK-3D U-Net + Tversky]")
        self.logger.info(f"Best Mean DSC : {best_dsc:.4f}  (epoch {best_epoch_r['epoch']+1})")
        self.logger.info(f"Total time    : {total_time/3600:.2f} hours")
        if self.inference_time_s is not None:
            self.logger.info(f"Inference     : {self.inference_time_s*1000:.1f} ms/volume")
        self.logger.info("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Proposed: MMSK-3D U-Net + Tversky Loss (Full Method)"
    )
    parser.add_argument("--gpu",          type=int,   default=0)
    parser.add_argument("--dataset_json", type=str,   required=True)
    parser.add_argument("--output_dir",   type=str,
                        default="./output_proposed_mmsk_tversky")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--vram_gb",      type=float, default=20.0)
    args = parser.parse_args()

    with open(args.dataset_json) as f:
        dataset = json.load(f)

    trainer = Trainer(args.gpu, args.output_dir, vram_gb=args.vram_gb)
    trainer.run(dataset["train"], dataset["val"], args.epochs)
