"""
Prepare dataset.json for BraTS 2024 Synapse GLI
from the folder structure:
    braTS/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2/
        BraTS-GLI-XXXXX-YYY/
            *-t1n.nii.gz   (T1)
            *-t1c.nii.gz   (T1ce)
            *-t2w.nii.gz   (T2)
            *-t2f.nii.gz   (FLAIR)
            *-seg.nii.gz   (label)

Split: 70% train / 15% val / 15% test  (stratified, reproducible)

Usage:
    python prepare_dataset.py \
        --data_dir ~/arsyadl/braTS/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2 \
        --output    ~/arsyadl/mmsk/dataset.json \
        --seed 42
"""

import json
import random
import argparse
from pathlib import Path


def main(args):
    data_dir = Path(args.data_dir).expanduser()
    output   = Path(args.output).expanduser()

    # Collect all case folders that have ALL 5 files
    cases = []
    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        name   = case_dir.name                              # e.g. BraTS-GLI-00001-000
        t1n    = case_dir / f"{name}-t1n.nii.gz"
        t1c    = case_dir / f"{name}-t1c.nii.gz"
        t2w    = case_dir / f"{name}-t2w.nii.gz"
        t2f    = case_dir / f"{name}-t2f.nii.gz"
        seg    = case_dir / f"{name}-seg.nii.gz"

        if not all(f.exists() for f in [t1n, t1c, t2w, t2f, seg]):
            print(f"  [SKIP] Missing files in {name}")
            continue

        # MONAI multi-channel image: stack [T1, T1ce, T2, FLAIR] as channels
        # in_channels=4 order must match model expectation
        cases.append({
            "image": [str(t1n), str(t1c), str(t2w), str(t2f)],
            "label": str(seg),
        })

    print(f"Found {len(cases)} valid cases with ground truth")

    # Reproducible shuffle & split
    random.seed(args.seed)
    random.shuffle(cases)

    n       = len(cases)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    # rest → test

    train = cases[:n_train]
    val   = cases[n_train : n_train + n_val]
    test  = cases[n_train + n_val :]

    print(f"Split  →  train: {len(train)} | val: {len(val)} | test: {len(test)}")

    dataset = {
        "train": train,
        "val":   val,
        "test":  test,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="~/arsyadl/braTS/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2")
    parser.add_argument("--output",   type=str,
                        default="~/arsyadl/mmsk/dataset.json")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    main(args)
