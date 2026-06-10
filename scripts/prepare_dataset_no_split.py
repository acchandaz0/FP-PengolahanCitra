"""
Prepare dataset.json for MMSK 3D U-Net

BraTS sudah menyediakan split terpisah:
- Training: /home/mci/arsyadl/braTS/training_data1_v2
- Validation: /home/mci/arsyadl/braTS/validation_data

Script ini hanya membuat dataset.json yang menunjuk ke preprocessed data.

Usage:
    python prepare_dataset_no_split.py
"""

import json
from pathlib import Path

# Paths
PREPROCESSED_TRAIN = Path("/home/mci/arsyadl/braTS/preprocessed")
PREPROCESSED_VAL = Path("/home/mci/arsyadl/braTS/preprocessed_test")
OUTPUT_JSON = Path("/home/mci/arsyadl/mmsk/dataset.json")

def collect_samples(preprocessed_dir):
    """Collect all .npz files from preprocessed directory"""
    if not preprocessed_dir.exists():
        print(f"⚠️  Directory not found: {preprocessed_dir}")
        return []
    
    samples = sorted([str(f) for f in preprocessed_dir.glob("*.npz")])
    return samples

def main():
    print("="*60)
    print("Preparing dataset.json (NO SPLIT)")
    print("="*60)
    
    # Collect training samples
    print(f"\n📁 Scanning training data: {PREPROCESSED_TRAIN}")
    train_samples = collect_samples(PREPROCESSED_TRAIN)
    print(f"   Found: {len(train_samples)} samples")
    
    # Collect validation samples
    print(f"\n📁 Scanning validation data: {PREPROCESSED_VAL}")
    val_samples = collect_samples(PREPROCESSED_VAL)
    print(f"   Found: {len(val_samples)} samples")
    
    # Create dataset.json
    dataset = {
        "description": "BraTS 2024 Synapse GLI - MMSK 3D U-Net",
        "train": train_samples,
        "val": val_samples,
        "num_classes": 4,
        "labels": {
            "0": "Background",
            "1": "Whole Tumor (WT)",
            "2": "Tumor Core (TC)",
            "3": "Enhancing Tumor (ET)"
        },
        "modalities": ["T1", "T2", "FLAIR", "T1ce"],
        "input_shape": [4, 128, 128, 128],
        "note": "BraTS provides separate train/val splits - no additional split needed"
    }
    
    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✅ Dataset JSON saved to: {OUTPUT_JSON}")
    print(f"\n📊 Summary:")
    print(f"   Training samples: {len(train_samples)}")
    print(f"   Validation samples: {len(val_samples)}")
    print(f"   Total: {len(train_samples) + len(val_samples)}")
    
    if len(train_samples) == 0 or len(val_samples) == 0:
        print(f"\n⚠️  WARNING: Some splits are empty!")
        print(f"   Make sure you've run preprocessing for both:")
        print(f"   1. Training data → {PREPROCESSED_TRAIN}")
        print(f"   2. Validation data → {PREPROCESSED_VAL}")

if __name__ == "__main__":
    main()
