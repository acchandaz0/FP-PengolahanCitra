"""
Baseline 1: Standard 3D U-Net + Dice Loss

Metrics per epoch: DSC, HD95, Sensitivity, Precision  (per class: NCR/ED/NET/ET)
FLOPs: computed once at startup.
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
    BraTSNpzDataset, validate_npz_files, setup_logger,
    compute_metrics, count_flops, log_epoch, make_epoch_record,
    OUT_CHANNELS, FG_CLASS_NAMES,
)


class Trainer:
    def __init__(self, gpu_id: int, output_dir: str):
        self.output_dir = Path(output_dir)
        self.logger     = setup_logger("baseline1", self.output_dir)
        self.device     = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"Device       : {self.device}")
        self.logger.info(f"Model        : Standard 3D U-Net (MONAI)")
        self.logger.info(f"Loss         : Dice  [Baseline 1]")
        self.logger.info(f"out_channels : {OUT_CHANNELS}  labels {{0,1,2,3,4}}")

        self.model = UNet(
            spatial_dims=3, in_channels=4, out_channels=OUT_CHANNELS,
            channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=2,
        ).to(self.device)

        self.loss_fn   = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)

        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Parameters   : {total:,}")
        self.flops_g = count_flops(self.model, device=self.device, logger=self.logger)

    # ──────────────────────────────────────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(inputs), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                self.logger.debug(f"  batch {batch_idx:>4} loss={loss.item():.4f}")
        return total_loss / len(loader)

    # ──────────────────────────────────────────────────────────────────────────
    def validate(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        val_loss = 0.0
        # Accumulate predictions and labels across batches for metric computation
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                inputs  = batch["image"].to(self.device)
                labels  = batch["label"].to(self.device)
                logits  = self.model(inputs)
                val_loss += self.loss_fn(logits, labels).item()
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        # Compute metrics on CPU to save GPU memory
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        m = compute_metrics(logits_cat, labels_cat)
        return val_loss / len(loader), m

    # ──────────────────────────────────────────────────────────────────────────
    def run(self, train_files: list, val_files: list, epochs: int = 300):
        train_files = validate_npz_files(train_files, self.logger)
        val_files   = validate_npz_files(val_files,   self.logger)
        if not train_files or not val_files:
            raise RuntimeError("No valid files remain after validation scan.")

        train_loader = DataLoader(
            BraTSNpzDataset(train_files, augment=True),
            batch_size=2, shuffle=True, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            BraTSNpzDataset(val_files, augment=False),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
        )
        self.logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        self.logger.info("=" * 65)

        results, best_dsc = [], 0.0
        start = time.time()

        for epoch in range(epochs):
            t0 = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            train_loss          = self.train_epoch(train_loader)
            val_loss, m         = self.validate(val_loader)
            self.scheduler.step()

            epoch_time = time.time() - t0
            elapsed    = time.time() - start
            lr         = self.optimizer.param_groups[0]["lr"]

            log_epoch(self.logger, epoch, epochs, train_loss, val_loss, m,
                      epoch_time, elapsed, lr)

            rec = make_epoch_record(epoch, train_loss, val_loss, m, lr,
                                    epoch_time, elapsed)
            results.append(rec)

            if m["dsc_mean"] > best_dsc:
                best_dsc = m["dsc_mean"]
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
                self.logger.info(f"  ✓ Best model saved (DSC={best_dsc:.4f})")

            if (epoch + 1) % 50 == 0:
                ckpt = self.output_dir / f"checkpoint_epoch{epoch+1}.pth"
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_dsc": best_dsc}, ckpt)
                self.logger.info(f"  Checkpoint → {ckpt.name}")

            self.logger.info("-" * 65)

        best_r = max(results, key=lambda x: x["dsc_mean"])
        with open(self.output_dir / "results.json", "w") as f:
            json.dump({
                "model": "Baseline1_3DUNet_Dice",
                "loss": "Dice", "architecture": "Standard 3D U-Net (MONAI)",
                "out_channels": OUT_CHANNELS, "fg_class_names": FG_CLASS_NAMES,
                "flops_giga": self.flops_g,
                "ablation_role": "Performance floor",
                "best_dsc": best_dsc, "best_epoch": best_r["epoch"] + 1,
                "best_dsc_per_class": best_r["dsc"],
                "best_hd95_per_class": best_r["hd95"],
                "best_sensitivity_per_class": best_r["sensitivity"],
                "best_precision_per_class": best_r["precision"],
                "total_time_hours": (time.time() - start) / 3600,
                "results": results,
            }, f, indent=2)

        self.logger.info("=" * 65)
        self.logger.info(f"Done — Best Mean DSC: {best_dsc:.4f} (epoch {best_r['epoch']+1})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",          type=int, default=0)
    p.add_argument("--dataset_json", type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="./output_baseline1")
    p.add_argument("--epochs",       type=int, default=300)
    args = p.parse_args()
    with open(args.dataset_json) as f:
        ds = json.load(f)
    Trainer(args.gpu, args.output_dir).run(ds["train"], ds["val"], args.epochs)