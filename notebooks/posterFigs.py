# %%
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.measure import label
from skimage.color import label2rgb

from skimage.draw import polygon2mask
from skimage.draw import polygon
# %%
datasetDicts = np.load('../data/TJ2201/split16/TJ2201DatasetDict.npy', allow_pickle=True)
# %%
# Find images with only one cell
oneCell = []
for seg in datasetDicts:
    if len(seg['annotations']) > 10:
        oneCell.append(seg)

# %%
idx = 2 # '../data/TJ2201/split16/phaseContrast/phaseContrast_E7_6_2022y04m07d_20h00m_11.png'
seg = oneCell[idx]

filePath = Path(seg['file_name'])
# Get rid of relative portions
filePath =  '../' / Path(*filePath.parts[2:])
img = imread(filePath)
plt.figure(figsize=(20,20/3))

plt.subplot(131)
plt.imshow(imread(str(filePath).replace('phaseContrast', 'composite')))
plt.axis('off')

# Make mask
plt.subplot(132)
plt.imshow(img)
for annotation in seg['annotations']:
    polyx = annotation['segmentation'][0][::2]
    polyy = annotation['segmentation'][0][1::2]

    poly = np.array([[y,x] for x,y in zip(polyx, polyy)])
    plt.plot(polyx, polyy)


plt.axis('off')

mask = np.zeros(np.shape(img))
for annotation in seg['annotations']:
    polyx = annotation['segmentation'][0][::2]
    polyy = annotation['segmentation'][0][1::2]

    poly = np.array([[y,x] for x,y in zip(polyx, polyy)])
    # plt.plot(polyx, polyy)


    rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
    if annotation['category_id'] == 1:
        img[rr, cc, :] = (255,0,0)
    else:
        img[rr, cc, :] = (0,255,0)

plt.subplot(133)
plt.imshow(img)
plt.axis('off')

plt.savefig('../figures/temp/pipeline.png', dpi=500)