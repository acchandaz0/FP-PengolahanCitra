"""
Baseline 1: Standard 3D U-Net + Dice Loss

Ablation Study Role:
    Baseline 1 (3D U-Net + Dice)  vs  Baseline 2 (3D U-Net + Tversky)
    → Isolates loss function contribution; architecture unchanged.

    Baseline 1 (3D U-Net + Dice)  vs  Ablation 1 (SK + Tversky)
    → Isolates SK architecture contribution against plain U-Net floor.

    Baseline 1 vs Proposed (MMSK + Tversky)
    → Full system gain (architecture + loss combined).

BUGFIX vs the previous version
------------------------------
The previous train_epoch() and validate() each ran the forward pass + loss
TWICE (two identical `with autocast(...)` blocks back-to-back). That:
    * doubled compute (slower epochs), and
    * in train_epoch, built the graph twice while backprop/accounting used a
      mix of the two — corrupting the gradient step and the logged loss.
Each now runs the forward pass exactly ONCE. If your existing B1 checkpoint
was trained with the buggy version, retrain or at least treat its collapse
with caution (it may be partly an artifact of the double-forward, not purely
a Dice-loss collapse).

Metrics recorded per epoch:
    DSC, HD95, Sensitivity, Precision per class (NCR/ED/NET/ET)
Also recorded in results.json:
    FLOPs (GFLOPs), parameter count, inference time, training time
"""

import torch
import json
import time
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss

from brats_utils import (
    BraTSNpzDataset,
    validate_npz_files,
    setup_logger,
    count_flops,
    StreamingMetrics,
    OUT_CHANNELS,
    FG_CLASS_NAMES,
)


class Trainer:
    def __init__(self, gpu_id: int, output_dir: str, vram_gb: float = 20.0):
        self.output_dir = Path(output_dir)
        self.logger     = setup_logger("baseline1", self.output_dir)
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
        self.logger.info(f"Model          : Standard 3D U-Net (MONAI)")
        self.logger.info(f"Loss           : Dice  [Baseline 1]")
        self.logger.info(f"out_channels   : {OUT_CHANNELS}")
        self.logger.info(f"Label space    : BraTS 2024 raw {{0,1,2,3,4}}")
        self.logger.info(f"Output dir     : {self.output_dir}")

        # ── Model ──────────────────────────────────────────────────────────
        self.model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=OUT_CHANNELS,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2,
        ).to(self.device)

        # ── Loss ───────────────────────────────────────────────────────────
        self.loss_fn = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
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

        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total:,}")
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

            # guard label: cegah index-out-of-bounds di one_hot/scatter_
            labels = labels.long().clamp_(0, 4)

            # ── single forward + loss (no duplicate block) ──
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
        Evaluate on validation set using StreamingMetrics (accumulates
        TP/FP/FN per batch, no CPU memory spike from stacking all logits).
        """
        self.model.eval()
        val_loss = 0.0
        sm = StreamingMetrics(num_classes=OUT_CHANNELS)

        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # guard label: cegah index-out-of-bounds di one_hot/scatter_
                labels = labels.long().clamp_(0, 4)

                # ── single forward + loss (no duplicate block) ──
                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    logits = self.model(inputs)
                    loss   = self.loss_fn(logits, labels)

                val_loss += loss.item()
                sm.add_batch(logits.cpu(), labels.cpu())

        m = sm.compute()

        # ── Inference timing (first epoch only) ────────────────────────────
        if self.inference_time_s is None:
            torch.cuda.empty_cache()
            dummy = torch.zeros(1, 4, 128, 128, 128, device=self.device)
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(dummy)
                torch.cuda.synchronize()
                n_timed = 10
                t0 = time.time()
                for _ in range(n_timed):
                    _ = self.model(dummy)
                torch.cuda.synchronize()
            self.inference_time_s = (time.time() - t0) / n_timed
            self.logger.info(
                f"Inference time: {self.inference_time_s * 1000:.1f} ms/volume "
                f"(avg over {n_timed} passes)"
            )
            del dummy
            torch.cuda.empty_cache()

        return val_loss / len(loader), m

    # ──────────────────────────────────────────────────────────────────────────
    def run(self, train_files: list, val_files: list,
            epochs: int = 300, resume_path: str | None = None,
            num_workers: int = 4):

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
                num_workers=num_workers, pin_memory=(num_workers > 0),
                drop_last=True,
                persistent_workers=(num_workers > 0),
            )
            vl = DataLoader(
                val_ds, batch_size=1, shuffle=False,
                num_workers=num_workers, pin_memory=(num_workers > 0),
                persistent_workers=(num_workers > 0),
            )
            return tl, vl

        # OOM probe: try bs=4 first (plain U-Net is lighter than MMSK),
        # fall back to bs=2+accum=2 if needed
        batch_size, self.accum_steps = 4, 1
        self.logger.info(
            f"Probing batch_size={batch_size}, accum_steps={self.accum_steps} ..."
        )
        train_loader, val_loader = make_loaders(batch_size)

        self.model.train()
        try:
            _b = next(iter(train_loader))
            _labels = _b["label"].to(self.device).long().clamp_(0, 4)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                _out  = self.model(_b["image"].to(self.device))
                _loss = self.loss_fn(_out, _labels) / self.accum_steps
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
            batch_size, self.accum_steps = 2, 2
            train_loader, val_loader = make_loaders(batch_size)
            self.logger.warning(
                f"OOM at bs=4 — fallback: bs={batch_size}, "
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
        self.logger.info("=" * 65)

        results     = []
        best_dsc    = 0.0
        start_time  = time.time()
        start_epoch = 0

        # ── Resume from checkpoint ─────────────────────────────────────────
        if resume_path is not None:
            ckpt_path = Path(resume_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scaler_state_dict" in ckpt:
                    self.scaler.load_state_dict(ckpt["scaler_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                best_dsc    = ckpt.get("best_dsc", 0.0)
                start_epoch = ckpt.get("epoch", 0) + 1
                results     = ckpt.get("results", [])
            else:
                # bare state_dict (e.g. old best_model.pth from original script)
                self.model.load_state_dict(ckpt)
                self.logger.warning(
                    f"Loaded raw state_dict from {ckpt_path.name} — "
                    f"optimizer/scaler/scheduler not restored (epoch resets to 0)"
                )
            self.logger.info(
                f"Resumed from   : {ckpt_path.name} | "
                f"start_epoch={start_epoch} | best_dsc={best_dsc:.4f}"
            )

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs}")

            train_loss  = self.train_epoch(train_loader)

            torch.cuda.empty_cache()
            val_loss, m = self.validate(val_loader)
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            elapsed    = time.time() - start_time
            lr         = self.optimizer.param_groups[0]["lr"]

            done    = epoch - start_epoch + 1
            eta_sec = (elapsed / done) * (epochs - epoch - 1)
            eta_h   = int(eta_sec // 3600)
            eta_m   = int((eta_sec % 3600) // 60)

            dsc  = m["dsc"]
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

            results.append({
                "epoch":         epoch,
                "train_loss":    train_loss,
                "val_loss":      val_loss,
                "dsc_mean":      m["dsc_mean"],
                "dsc":           dsc,
                "hd95":          hd,
                "sensitivity":   sens,
                "precision":     prec,
                "lr":            lr,
                "epoch_time_s":  epoch_time,
                "elapsed_hours": elapsed / 3600,
            })

            # ── Save best model (full checkpoint, resume-compatible) ───────
            if m["dsc_mean"] > best_dsc:
                best_dsc = m["dsc_mean"]
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict":    self.scaler.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_dsc":             best_dsc,
                    "results":              results,
                }, self.output_dir / "best_model.pth")
                self.logger.info(f"  ✓ New best model saved (DSC: {best_dsc:.4f})")

            # ── Periodic checkpoint every 10 epochs ───────────────────────
            if (epoch + 1) % 10 == 0:
                ckpt = self.output_dir / f"checkpoint_epoch{epoch+1}.pth"
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict":    self.scaler.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_dsc":             best_dsc,
                    "results":              results,
                }, ckpt)
                self.logger.info(f"  Checkpoint saved → {ckpt.name}")

            self.logger.info("-" * 65)

        # ── Final summary ──────────────────────────────────────────────────
        total_time   = time.time() - start_time
        best_epoch_r = max(results, key=lambda x: x["dsc_mean"])

        with open(self.output_dir / "results.json", "w") as f:
            json.dump({
                "model":                      "Baseline1_3DUNet_Dice",
                "loss":                       "Dice",
                "architecture":               "Standard 3D U-Net (MONAI)",
                "ablation_role":              "Performance floor",
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

        self.logger.info("=" * 65)
        self.logger.info("Training complete! [Baseline 1: 3D U-Net + Dice]")
        self.logger.info(f"Best Mean DSC : {best_dsc:.4f}  (epoch {best_epoch_r['epoch']+1})")
        self.logger.info(f"Total time    : {total_time/3600:.2f} hours")
        if self.inference_time_s is not None:
            self.logger.info(f"Inference     : {self.inference_time_s*1000:.1f} ms/volume")
        self.logger.info("=" * 65)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Baseline 1: Standard 3D U-Net + Dice Loss"
    )
    p.add_argument("--gpu",          type=int,   default=0)
    p.add_argument("--dataset_json", type=str,   required=True)
    p.add_argument("--output_dir",   type=str,   default="./output_baseline1")
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--vram_gb",      type=float, default=20.0)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint .pth to resume from")
    p.add_argument("--num_workers",  type=int,   default=4,
                   help="DataLoader worker processes. Use 0 to disable "
                        "multiprocessing (safest on shared servers, "
                        "slower but eliminates segfault from worker crashes)")
    args = p.parse_args()

    with open(args.dataset_json) as f:
        ds = json.load(f)

    Trainer(args.gpu, args.output_dir, vram_gb=args.vram_gb).run(
        ds["train"], ds["val"], args.epochs,
        resume_path=args.resume,
        num_workers=args.num_workers,
    )