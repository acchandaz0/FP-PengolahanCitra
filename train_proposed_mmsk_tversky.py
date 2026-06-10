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
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.data import CacheDataset
from monai.transforms import (
    Compose, MapTransform, EnsureChannelFirstd, Spacingd,
    Orientationd, CropForegroundd, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, RandShiftIntensityd,
)
import argparse
import json
from pathlib import Path
import time
import numpy as np

# Import MMSK architecture
from mmsk_3d_unet import MMSK3DUNet


# Custom loader for .npz files
class LoadNPZd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            try:
                npz_data = np.load(d[key])
                if key == "image":
                    d[key] = np.array(npz_data["images"])  # (4, 128, 128, 128)
                elif key == "label":
                    d[key] = np.array(npz_data["seg"])  # (128, 128, 128)
                npz_data.close()
            except Exception as e:
                print(f"Error loading {d[key]}: {e}")
                raise
        return d


class Trainer:
    def __init__(self, gpu_id, output_dir):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using device: {self.device}")
        print(f"Model      : MMSK-3D U-Net (cross-modal gating)")
        print(f"Loss       : Tversky (α=0.3, β=0.7)   [PROPOSED]")

        # ── Full proposed architecture ─────────────────────────────────────
        self.model = MMSK3DUNet(
            in_channels=4,
            out_channels=5,
            store_attention=False,   # disable during training for speed
        ).to(self.device)

        # ── Tversky Loss ───────────────────────────────────────────────────
        # α=0.3 → lower FP penalty
        # β=0.7 → higher FN penalty (recall-focused, critical for tiny ET)
        self.loss_fn = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            alpha=0.3,
            beta=0.7,
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
            LoadNPZd(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
            ScaleIntensityRanged(keys=["image"], a_min=-3, a_max=3, b_min=0.0, b_max=1.0, clip=True),
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
        train_ds = CacheDataset(
            data=train_files,
            transform=self.get_transforms(train=True),
            cache_rate=0.5,
            num_workers=4,
        )
        val_ds = CacheDataset(
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
                "model": "Proposed_MMSK_Tversky",
                "loss": "Tversky(alpha=0.3, beta=0.7)",
                "architecture": "MMSK-3D U-Net (cross-modal gating)",
                "ablation_role": "Full proposed method — MMSK + Tversky combined",
                "results": results,
                "best_dice": best_metric,
                "total_time_hours": total_time / 3600,
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training completed! [Proposed: MMSK-3D U-Net + Tversky]")
        print(f"Best Dice : {best_metric:.4f}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Proposed: MMSK-3D U-Net + Tversky Loss (Full Method)"
    )
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default="./output_proposed_mmsk_tversky")
    parser.add_argument("--epochs",       type=int, default=300)
    args = parser.parse_args()

    with open(args.dataset_json) as f:
        dataset = json.load(f)

    # Validate files and skip corrupted ones
    print("Validating dataset files...")
    valid_train = []
    valid_val = []
    
    for fpath in dataset["train"]:
        try:
            data = np.load(fpath)
            _ = np.array(data["images"])
            _ = np.array(data["seg"])
            data.close()
            valid_train.append(fpath)
        except Exception as e:
            print(f"Skipping corrupted train file: {fpath}")
    
    for fpath in dataset["val"]:
        try:
            data = np.load(fpath)
            _ = np.array(data["images"])
            _ = np.array(data["seg"])
            data.close()
            valid_val.append(fpath)
        except Exception as e:
            print(f"Skipping corrupted val file: {fpath}")
    
    train_files = [{"image": f, "label": f} for f in valid_train]
    val_files   = [{"image": f, "label": f} for f in valid_val]
    print(f"Loaded dataset: {len(train_files)} train, {len(val_files)} val")

    trainer = Trainer(args.gpu, args.output_dir)
    trainer.run(train_files, val_files, args.epochs)
