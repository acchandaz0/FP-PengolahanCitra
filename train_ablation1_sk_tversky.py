"""
Ablation 1: SK-3D U-Net (BGSK) + Tversky Loss (α=0.3, β=0.7)

Ablation Study Role:
    Ablation 1 (SK + Tversky)  vs  Baseline 2 (3D U-Net + Tversky)
    → Isolates SK architecture contribution alone.

    Ablation 1 (SK + Tversky)  vs  Proposed (MMSK + Tversky)
    → Isolates cross-modal gating contribution (MMSK vs plain SK).

    Ablation 1 (SK + Tversky)  vs  Ablation 2 (MMSK + Dice)
    → Cross-comparison: single-stream SK+Tversky vs cross-modal gating+Dice.

Dataset:
    BraTS 2024 Synapse GLI, nnU-Net-style preprocessed .npz
    Labels (raw, no remap): 0=BG, 1=NCR, 2=ED, 3=NET, 4=ET
    out_channels = 5 (consistent with all other variants)

CHANGELOG vs original:
    [FIX] No remap in _load() — raw labels {0,1,2,3,4} used directly
    [FIX] Dice log: NCR/ED/NET/ET (4 foreground classes)
    [FIX] _parse_dice_scores() shared utility for robust extraction
    [KEEP] AMP, VRAM cap, OOM fallback, drop_last=True
"""

import torch
import json
import time
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric

from brats_dataset import (
    BraTSNpzDataset, validate_npz_files, setup_logger,
    OUT_CHANNELS, FG_CLASS_NAMES,
)
from bgsk_3d_unet_patched import BGSK3DUNet


class Trainer:
    def __init__(self, gpu_id: int, output_dir: str, vram_gb: float = 20.0):
        self.output_dir = Path(output_dir)
        self.logger     = setup_logger("ablation1", self.output_dir)
        self.device     = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        if self.device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            fraction   = min(vram_gb / total_vram, 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, device=gpu_id)
            self.logger.info(
                f"VRAM cap       : {vram_gb:.0f} GB / {total_vram:.0f} GB "
                f"(fraction={fraction:.3f})"
            )

        self.logger.info(f"Device         : {self.device}")
        self.logger.info(f"Model          : SK-3D U-Net (BGSK, no modal gate)")
        self.logger.info(f"Loss           : Tversky (α=0.3, β=0.7)  [Ablation 1]")
        self.logger.info(f"out_channels   : {OUT_CHANNELS}")
        self.logger.info(f"Label space    : BraTS 2024 raw {{0,1,2,3,4}}")
        self.logger.info(f"Output dir     : {self.output_dir}")

        # ── Model ──────────────────────────────────────────────────────────
        self.model = BGSK3DUNet(
            in_channels  = 4,
            out_channels = OUT_CHANNELS,
        ).to(self.device)

        # ── Loss ───────────────────────────────────────────────────────────
        self.loss_fn = TverskyLoss(
            to_onehot_y        = True,
            softmax            = True,
            alpha              = 0.3,
            beta               = 0.7,
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
        self.accum_steps = 2

        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total:,}")
        self.logger.info(f"AMP             : {'ON' if self.use_amp else 'OFF (CPU)'}")

    # ──────────────────────────────────────────────────────────────────────────
    def _parse_dice_scores(self, dice_scores) -> list[float]:
        raw = dice_scores.mean(dim=0) if dice_scores.dim() > 1 else dice_scores
        raw = raw.cpu().flatten().tolist()
        while len(raw) < 4:
            raw.append(float("nan"))
        return raw[:4]

    # ──────────────────────────────────────────────────────────────────────────
    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
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
        self.model.eval()
        self.metric.reset()
        val_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                inputs  = batch["image"].to(self.device)
                labels  = batch["label"].to(self.device)
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
                num_workers=2, pin_memory=True, drop_last=True,
            )
            vl = DataLoader(
                val_ds, batch_size=1, shuffle=False,
                num_workers=2, pin_memory=True,
            )
            return tl, vl

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
                f"effective_batch={batch_size*self.accum_steps}"
            )
        except _OOM:
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            batch_size, self.accum_steps = 1, 4
            train_loader, val_loader = make_loaders(batch_size)
            self.logger.warning(
                f"OOM at bs=2 — fallback: bs={batch_size}, "
                f"accum={self.accum_steps}"
            )

        self.logger.info(f"Train batches  : {len(train_loader)} (bs={batch_size})")
        self.logger.info(f"Val batches    : {len(val_loader)}")
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
                "model":               "Ablation1_SK_Tversky",
                "loss":                "Tversky(alpha=0.3, beta=0.7)",
                "architecture":        "SK-3D U-Net (BGSK, no modal gate)",
                "out_channels":        OUT_CHANNELS,
                "label_space":         "BraTS2024 raw {0,1,2,3,4}",
                "fg_class_names":      FG_CLASS_NAMES,
                "ablation_role":       "Isolates SK architecture vs Baseline 2",
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
        self.logger.info("Training complete! [Ablation 1: SK-3D U-Net + Tversky]")
        self.logger.info(f"Best Mean Dice : {best_metric:.4f}  (epoch {best_epoch_r['epoch']+1})")
        self.logger.info(f"Total time     : {total_time/3600:.2f} hours")
        self.logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation 1: SK-3D U-Net + Tversky Loss"
    )
    parser.add_argument("--gpu",          type=int,   default=0)
    parser.add_argument("--dataset_json", type=str,   required=True)
    parser.add_argument("--output_dir",   type=str,
                        default="./output_ablation1_sk_tversky")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--vram_gb",      type=float, default=20.0)
    args = parser.parse_args()

    import json as _json
    with open(args.dataset_json) as f:
        dataset = _json.load(f)

    trainer = Trainer(args.gpu, args.output_dir, vram_gb=args.vram_gb)
    trainer.run(dataset["train"], dataset["val"], args.epochs)