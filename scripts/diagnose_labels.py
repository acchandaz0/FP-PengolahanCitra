"""
Diagnostik label convention di preprocessed .npz files.
Jalankan di server: python diagnose_labels.py --dataset_json /path/to/dataset.json --n 30
"""
import numpy as np
import json
import argparse
from pathlib import Path
from collections import Counter

def diagnose(dataset_json, n_samples=30):
    with open(dataset_json) as f:
        ds = json.load(f)

    files = ds.get("train", []) + ds.get("val", [])
    files = files[:n_samples]

    print(f"Checking {len(files)} files from {dataset_json}\n")

    all_unique_labels = set()
    label_counts = Counter()
    shape_counter = Counter()
    bad_files = []

    for i, fp in enumerate(files):
        try:
            data = np.load(fp)
            seg = data["seg"]
            img = data["images"]
            unique = set(np.unique(seg).tolist())
            all_unique_labels |= unique
            for v in unique:
                label_counts[int(v)] += 1
            shape_counter[str(seg.shape)] += 1

            if i < 5:
                print(f"  [{i}] {Path(fp).name}")
                print(f"       images.shape = {img.shape}  dtype={img.dtype}")
                print(f"       seg.shape    = {seg.shape}  dtype={seg.dtype}")
                print(f"       seg unique   = {sorted(unique)}")
                print()
            data.close()
        except Exception as e:
            bad_files.append((fp, str(e)))
            print(f"  [ERR] {Path(fp).name}: {e}")

    print("="*55)
    print(f"ALL unique label values across {n_samples} files:")
    print(f"  {sorted(all_unique_labels)}")
    print()
    print(f"Files containing each label value:")
    for v in sorted(label_counts.keys()):
        print(f"  label={v}: present in {label_counts[v]}/{len(files)} files")
    print()
    print(f"seg shape distribution: {dict(shape_counter)}")
    print()

    # Diagnosis
    labels = sorted(all_unique_labels)
    print("DIAGNOSIS:")
    if set(labels) <= {0,1,2,3}:
        print("  → Labels {0,1,2,3}: 4-class remapped")
        print("     nnU-Net sudah remap: BG=0, NCR=1, ED=2, ET=3")
        print("     CORRECT: out_channels=4, no remap needed in _load()")
    elif set(labels) <= {0,1,2,4} or set(labels) <= {0,1,2,3,4}:
        print("  → Labels contain raw BraTS values (4 present)")
        if 3 in labels and 4 in labels:
            print("     BraTS 2024 raw: 0=BG,1=NCR,2=ED,3=NET,4=ET")
            print("     CORRECT: out_channels=5, no remap needed in _load()")
        elif 4 in labels and 3 not in labels:
            print("     BraTS classic: 0=BG,1=NCR,2=ED,4=ET (gap at 3)")
            print("     NEED REMAP: seg[seg==4]=3, out_channels=4")
    print()
    if bad_files:
        print(f"BAD FILES ({len(bad_files)}):")
        for fp, e in bad_files:
            print(f"  {fp}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    diagnose(args.dataset_json, args.n)