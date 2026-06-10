"""
Ablation 2: MMSK-3D U-Net + Dice Loss

Ablation Study Role:
    Ablation 2 (MMSK + Dice)  vs  Baseline 2 (3D U-Net + Tversky)
    → Separates gating contribution from loss effect.

    Ablation 2 (MMSK + Dice)  vs  Proposed (MMSK + Tversky)
    → Isolates the contribution of Tversky loss ON TOP of MMSK.

    Ablation 1 (SK + Tversky) vs  Ablation 2 (MMSK + Dice)
    → Cross-comparison: single-stream SK+loss vs cross-modal gating alone.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import CacheDataset
from robust_dataset import RobustCacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, CropForegroundd, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, RandShiftIntensityd,
)
import argparse
import json
from pathlib import Path
import time

# Import MMSK architecture
from mmsk_3d_unet import MMSK3DUNet


class Trainer:
    def __init__(self, gpu_id, output_dir):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using device: {self.device}")
        print(f"Model      : MMSK-3D U-Net (cross-modal gating)")
        print(f"Loss       : Dice   [Ablation 2]")

        # ── MMSK architecture — cross-modal gating active ─────────────────
        self.model = MMSK3DUNet(
            in_channels=4,
            out_channels=4,
            store_attention=False,   # disable during training for speed
        ).to(self.device)

        # ── Dice Loss (symmetric) — standard baseline loss ─────────────────
        self.loss_fn = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
        )

        self.metric    = DiceMetric(include_background=False, reduction="mean_batch")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

    # ──────────────────────────────────────────────────────────────────────
    def get_transforms(self, train=True):
        t = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1, b_min=0.0, b_max=1.0, clip=True),
        ]
        if train:
            t.extend([
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            ])
        return Compose(t)

    # ──────────────────────────────────────────────────────────────────────
    def train_epoch(self, loader, epoch):
        self.model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(loader):
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss    = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")

        return epoch_loss / len(loader)

    # ──────────────────────────────────────────────────────────────────────
    def validate(self, loader):
        self.model.eval()
        self.metric.reset()
        val_loss = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs  = self.model(inputs)
                loss     = self.loss_fn(outputs, labels)
                val_loss += loss.item()

                self.metric(y_pred=outputs, y=labels)

        dice_scores = self.metric.aggregate()
        return val_loss / len(loader), dice_scores

    # ──────────────────────────────────────────────────────────────────────
    def run(self, train_files, val_files, epochs=300):
        print(f"\nPreparing datasets...")
        train_ds = RobustCacheDataset(
            data=train_files,
            transform=self.get_transforms(train=True),
            cache_rate=0.5,
            num_workers=4,
        )
        val_ds = RobustCacheDataset(
            data=val_files,
            transform=self.get_transforms(train=False),
            cache_rate=1.0,
            num_workers=4,
        )

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        results     = []
        best_metric = 0
        start_time  = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss            = self.train_epoch(train_loader, epoch)
            val_loss, dice_scores = self.validate(val_loader)
            self.scheduler.step()

            mean_dice  = dice_scores.mean().item()
            epoch_time = time.time() - epoch_start

            results.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice_mean": mean_dice,
                "val_dice_per_class": dice_scores.mean(dim=0).cpu().tolist(),
                "lr": self.optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time,
            })

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Mean Dice: {mean_dice:.4f}")
            print(f"  Dice per class (NCR/ED/ET): {dice_scores.mean(dim=0).cpu().tolist()}")
            print(f"  Time: {epoch_time:.1f}s | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if mean_dice > best_metric:
                best_metric = mean_dice
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
                print(f"  ✓ New best model saved (Dice: {best_metric:.4f})")

            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_metric': best_metric,
                }, self.output_dir / f"checkpoint_epoch{epoch+1}.pth")

        total_time = time.time() - start_time

        with open(self.output_dir / "results.json", "w") as f:
            json.dump({
                "model": "Ablation2_MMSK_Dice",
                "loss": "Dice",
                "architecture": "MMSK-3D U-Net (cross-modal gating)",
                "ablation_role": "Isolates cross-modal gating contribution (no Tversky)",
                "results": results,
                "best_dice": best_metric,
                "total_time_hours": total_time / 3600,
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training completed! [Ablation 2: MMSK-3D U-Net + Dice]")
        print(f"Best Dice : {best_metric:.4f}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation 2: MMSK-3D U-Net + Dice Loss"
    )
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default="./output_ablation2_mmsk_dice")
    parser.add_argument("--epochs",       type=int, default=300)
    args = parser.parse_args()

    with open(args.dataset_json) as f:
        dataset = json.load(f)

    train_files = dataset["train"]
    val_files   = dataset["val"]
    print(f"Loaded dataset: {len(train_files)} train, {len(val_files)} val")

    trainer = Trainer(args.gpu, args.output_dir)
    trainer.run(train_files, val_files, args.epochs)
