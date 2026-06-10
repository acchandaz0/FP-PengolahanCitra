"""
robust_dataset.py
Utility untuk skip file corrupt secara otomatis saat loading dataset.

Digunakan oleh semua training script sebagai pengganti CacheDataset biasa.
"""

import gzip
import torch
from torch.utils.data import Dataset as TorchDataset
from monai.data import CacheDataset
from monai.transforms import Compose


def is_valid_nifti(file_entry):
    """
    Cek apakah file .nii.gz bisa dibaca (tidak corrupt).
    file_entry bisa berupa string path atau list of string paths.
    """
    paths = file_entry if isinstance(file_entry, list) else [file_entry]
    for path in paths:
        try:
            with gzip.open(path, 'rb') as gz:
                while gz.read(65536):
                    pass
        except Exception:
            return False
    return True


def filter_corrupt(data_list, verbose=True):
    """
    Filter list of dicts (MONAI format) dan buang entry yang file-nya corrupt.

    Args:
        data_list : list of {"image": [...], "label": "..."}
        verbose   : print nama file yang di-skip

    Returns:
        clean_list : list tanpa entry corrupt
    """
    clean = []
    skipped = 0

    for entry in data_list:
        image_ok = is_valid_nifti(entry["image"])
        label_ok = is_valid_nifti(entry["label"])

        if image_ok and label_ok:
            clean.append(entry)
        else:
            skipped += 1
            if verbose:
                label_str = entry["label"] if isinstance(entry["label"], str) else entry["label"]
                # Ambil nama case dari path label
                import pathlib
                case_name = pathlib.Path(label_str).parent.name
                print(f"  [SKIP corrupt] {case_name}")

    if skipped > 0:
        print(f"\n  → Skipped {skipped} corrupt case(s), using {len(clean)} clean cases.\n")

    return clean


class RobustCacheDataset(CacheDataset):
    """
    CacheDataset yang otomatis skip file corrupt sebelum caching.
    Drop-in replacement untuk CacheDataset di semua training script.
    """
    def __init__(self, data, transform, cache_rate=0.5, num_workers=4, verbose=True):
        print(f"  Validating {len(data)} files for corruption...")
        clean_data = filter_corrupt(data, verbose=verbose)
        super().__init__(
            data=clean_data,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
