# %%
from src.data.imageProcessing import bbIncrease

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from skimage.io import imread
# %%
homePath = Path('../../../')
datasetDictPath = homePath / './data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy'
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
# %% Get single cell

for seg in datasetDicts:
    annotations = seg['annotations']
    if len(annotations) <= 0:
        continue
    imgPath = homePath / Path(*Path(seg['file_name']).parts[2:])

    img = imread(imgPath)

    cell = annotations[0]
    break
polyx = cell['segmentation'][0][::2]
polyy = cell['segmentation'][0][1::2]

plt.imshow(img)
plt.plot(polyx, polyy, linewidth=3)
# %%


# %%