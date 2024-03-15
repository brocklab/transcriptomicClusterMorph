import numpy as np
import os
files = os.listdir('.')
c = 0
for file in files:
    if not file.endswith('.npy'):
        continue
    x = np.load(file, allow_pickle = True)
    x = x.item()
    c += len(np.unique(x['masks']))-1

print(f'Found {c} outlines')
