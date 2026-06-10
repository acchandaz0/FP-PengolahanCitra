"""
Ablation 2: MMSK-3D U-Net + Dice Loss

Ablation Study Role:
    Ablation 2 (MMSK + Dice)  vs  Baseline 2 (3D U-Net + Tversky)
    → Separates cross-modal gating contribution from loss effect.

    Ablation 2 (MMSK + Dice)  vs  Proposed (MMSK + Tversky)
    → Isolates Tversky loss contribution on top of MMSK.

    Ablation 1 (SK + Tversky) vs  Ablation 2 (MMSK + Dice)
    → Cross-comparison: single-stream SK+Tversky vs cross-modal gating+Dice.

Dataset:
    BraTS 2024 Synapse GLI, nnU-Net-style preprocessed .npz
    Labels (raw, no remap): 0=BG, 1=NCR, 2=ED, 3=NET, 4=ET
    out_channels = 5 (consistent with all other variants)

CHANGELOG vs original (which was broken — Dice stuck at 0.01 for 300 epochs):
    [FIX] ROOT CAUSE: _load() was remapping seg[seg==4]=3 making label range [0,3],
          but out_channels=5 expected [0,4]. Model predicting class 4 which never
          exists in labels → DiceMetric always 0 for that class → mean stuck.
          FIX: Removed all remap logic. Raw labels {0,1,2,3,4} used directly.
    [FIX] Removed erroneous seg[seg==3]=2 remap (was treating NET as ED)
    [FIX] defensive clamp now clamp(0,4) not clamp(0,3)
    [FIX] train_epoch defensive clamp: clamp(0,4) not clamp(0,4) with wrong comment
    [FIX] validate() defensive clamp consistent with train_epoch
    [FIX] Dice log: NCR/ED/NET/ET (4 foreground classes)
    [KEEP] AMP, OOM fallback (bs=4 → bs=2+accum=2), drop_last=True
"""

import torch
import json
import time
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

from brats_dataset import (
    BraTSNpzDataset, validate_npz_files, setup_logger,
    OUT_CHANNELS, FG_CLASS_NAMES,
)
from mmsk_3d_unet import MMSK3DUNet


class Trainer:
    def __init__(self, gpu_id: int, output_dir: str):
        self.output_dir = Path(output_dir)
        self.logger     = setup_logger("ablation2", self.output_dir)
        self.device     = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        self.logger.info(f"Device         : {self.device}")
        self.logger.info(f"Model          : MMSK-3D U-Net (cross-modal gating)")
        self.logger.info(f"Loss           : Dice  [Ablation 2]")
        self.logger.info(f"out_channels   : {OUT_CHANNELS}")
        self.logger.info(f"Label space    : BraTS 2024 raw {{0,1,2,3,4}}")
        self.logger.info(f"Output dir     : {self.output_dir}")

        # ── Model ──────────────────────────────────────────────────────────
        self.model = MMSK3DUNet(
            in_channels    = 4,
            out_channels   = OUT_CHANNELS,
            store_attention = False,
        ).to(self.device)

        # ── Loss ───────────────────────────────────────────────────────────
        # Standard symmetric Dice — no recall/precision asymmetry.
        # Contrast with Proposed (Tversky α=0.3,β=0.7) to isolate loss effect.
        self.loss_fn = DiceLoss(
            to_onehot_y        = True,
            softmax            = True,
            include_background = False,
        )

        # ── Metric / Optimizer / Scheduler ─────────────────────────────────
        self.metric    = DiceMetric(include_background=False, reduction="mean_batch")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=300
        )

        # ── AMP ────────────────────────────────────────────────────────────
        self.use_amp     = self.device.type == "cuda"
        self.scaler      = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)
        self.accum_steps = 1    # set in run() after OOM probe

        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total:,}")
        self.logger.info(f"AMP             : {'ON' if self.use_amp else 'OFF (CPU)'}")

    # ──────────────────────────────────────────────────────────────────────────
    def _parse_dice_scores(self, dice_scores) -> list[float]:
        raw = dice_scores.mean(dim=0) if dice_scores.dim() > 1 else dice_scores
        raw = raw.cpu().flatten().tolist()
        while len(raw) < 4:
            raw.append(float("nan"))
        return raw[:4]   # [NCR, ED, NET, ET]

    # ──────────────────────────────────────────────────────────────────────────
    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Defensive clamp — no-op on healthy data; protects against any
            # corrupt patch that slipped past pre-flight validation.
            # Must match OUT_CHANNELS=5 → valid class index range [0,4].
            labels = labels.clamp(0, 4)

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
        self.model.eval()
        self.metric.reset()
        val_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                inputs  = batch["image"].to(self.device)
                labels  = batch["label"].to(self.device)
                labels  = labels.clamp(0, 4)   # consistent with train_epoch

                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss    = self.loss_fn(outputs, labels)
                val_loss += loss.item()
                self.metric(y_pred=outputs, y=labels)

        return val_loss / len(loader), self.metric.aggregate()

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

        # Try bs=4 first (MMSK is heavier than plain U-Net, but RTX 8000 has headroom)
        # Fall back to bs=2+accum=2 if OOM
        batch_size, self.accum_steps = 4, 1
        self.logger.info(f"Probing batch_size={batch_size} ...")
        train_loader, val_loader = make_loaders(batch_size)

        self.model.train()
        try:
            _b = next(iter(train_loader))
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                _out  = self.model(_b["image"].to(self.device))
                _loss = self.loss_fn(_out, _b["label"].to(self.device).clamp(0,4))
            self.scaler.scale(_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            self.logger.info(
                f"Probe OK — bs={batch_size}, accum={self.accum_steps}, "
                f"effective_batch={batch_size*self.accum_steps}"
            )
        except _OOM:
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            batch_size, self.accum_steps = 2, 2
            train_loader, val_loader = make_loaders(batch_size)
            self.logger.warning(
                f"OOM at bs=4 — fallback: bs={batch_size}, "
                f"accum={self.accum_steps}, "
                f"effective_batch={batch_size*self.accum_steps}"
            )

        self.logger.info(f"Train batches  : {len(train_loader)} (bs={batch_size})")
        self.logger.info(f"Val batches    : {len(val_loader)}")
        self.logger.info(
            f"Accum steps    : {self.accum_steps} "
            f"(effective batch={batch_size*self.accum_steps})"
        )
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info("=" * 60)

        results, best_metric = [], 0.0
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs}")

            train_loss            = self.train_epoch(train_loader, epoch)
            val_loss, dice_scores = self.validate(val_loader)
            self.scheduler.step()

            mean_dice  = dice_scores.mean().item()
            epoch_time = time.time() - epoch_start
            elapsed    = time.time() - start_time
            lr         = self.optimizer.param_groups[0]["lr"]
            per_class  = self._parse_dice_scores(dice_scores)

            eta_sec = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            eta_h, eta_m = int(eta_sec // 3600), int((eta_sec % 3600) // 60)

            results.append({
                "epoch":              epoch,
                "train_loss":         train_loss,
                "val_loss":           val_loss,
                "val_dice_mean":      mean_dice,
                "val_dice_per_class": per_class,
                "lr":                 lr,
                "epoch_time":         epoch_time,
                "elapsed_hours":      elapsed / 3600,
            })

            self.logger.info(
                f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Mean Dice: {mean_dice:.4f}"
            )
            self.logger.info(
                f"  Dice  NCR/ED/NET/ET: "
                f"{per_class[0]:.4f} / {per_class[1]:.4f} / "
                f"{per_class[2]:.4f} / {per_class[3]:.4f}"
            )
            self.logger.info(
                f"  Time: {epoch_time:.1f}s | Elapsed: {elapsed/3600:.2f}h | "
                f"ETA: {eta_h}h {eta_m}m | LR: {lr:.2e}"
            )

            if mean_dice > best_metric:
                best_metric = mean_dice
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "best_model.pth",
                )
                self.logger.info(f"  ✓ New best model saved (Dice: {best_metric:.4f})")

            if (epoch + 1) % 50 == 0:
                ckpt = self.output_dir / f"checkpoint_epoch{epoch+1}.pth"
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_metric":          best_metric,
                }, ckpt)
                self.logger.info(f"  Checkpoint saved → {ckpt.name}")

            self.logger.info("-" * 60)

        total_time   = time.time() - start_time
        best_epoch_r = max(results, key=lambda x: x["val_dice_mean"])

        import json as _json
        with open(self.output_dir / "results.json", "w") as f:
            _json.dump({
                "model":               "Ablation2_MMSK_Dice",
                "loss":                "Dice",
                "architecture":        "MMSK-3D U-Net (cross-modal gating)",
                "out_channels":        OUT_CHANNELS,
                "label_space":         "BraTS2024 raw {0,1,2,3,4}",
                "fg_class_names":      FG_CLASS_NAMES,
                "ablation_role":       "Isolates cross-modal gating (no Tversky)",
                "batch_size":          batch_size,
                "accum_steps":         self.accum_steps,
                "effective_batch":     batch_size * self.accum_steps,
                "best_dice":           best_metric,
                "best_epoch":          best_epoch_r["epoch"] + 1,
                "best_dice_per_class": best_epoch_r["val_dice_per_class"],
                "total_time_hours":    total_time / 3600,
                "results":             results,
            }, f, indent=2)

        self.logger.info("=" * 60)
        self.logger.info("Training complete! [Ablation 2: MMSK-3D U-Net + Dice]")
        self.logger.info(f"Best Mean Dice : {best_metric:.4f}  (epoch {best_epoch_r['epoch']+1})")
        self.logger.info(f"Total time     : {total_time/3600:.2f} hours")
        self.logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation 2: MMSK-3D U-Net + Dice Loss"
    )
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--output_dir",   type=str,
                        default="./output_ablation2_mmsk_dice")
    parser.add_argument("--epochs",       type=int, default=300)
    args = parser.parse_args()

    import json as _json
    with open(args.dataset_json) as f:
        dataset = _json.load(f)

    trainer = Trainer(args.gpu, args.output_dir)
    trainer.run(dataset["train"], dataset["val"], args.epochs)