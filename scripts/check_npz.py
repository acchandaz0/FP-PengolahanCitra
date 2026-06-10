import numpy as np
import json
from pathlib import Path

with open('dataset.json') as f:
    dataset = json.load(f)

corrupted = []
all_files = dataset['train'] + dataset['val']
for i, fpath in enumerate(all_files):
    try:
        data = np.load(fpath)
        images = np.array(data['images'])
        seg = np.array(data['seg'])
        data.close()
        if i % 100 == 0:
            print(f"Checked {i}/{len(all_files)}...")
    except Exception as e:
        print(f"Corrupted: {fpath} - {e}")
        corrupted.append(fpath)

print(f"\nTotal corrupted: {len(corrupted)}")
if corrupted:
    print("\nRemoving from dataset.json...")
    dataset['train'] = [f for f in dataset['train'] if f not in corrupted]
    dataset['val'] = [f for f in dataset['val'] if f not in corrupted]
    
    with open('dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"New counts: {len(dataset['train'])} train, {len(dataset['val'])} val")
