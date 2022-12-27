# %% [markdown]
"""
Connects split image coordinates to whole image coordinates
"""
# %%
import numpy as np
import os 
import matplotlib.pyplot as plt

from skimage.io import imread
# %%
def splitName2Whole(imgName):
    extSplit = imgName.split('.')
    ext = extSplit[-1]
    imgName = extSplit[0]

    imgName = '_'.join(imgName.split('_')[0:-1])+'.'+ext
    return imgName
# %%
idx = 2
splitDir = '../../data/TJ2201/TJ2201Split16/phaseContrast'
wholeDir = '../../data/TJ2201/TJ2201Raw/phaseContrast'

imgs = os.listdir(splitDir)

splitImgName = imgs[idx]
wholeImgName = splitName2Whole(splitImgName)
splitImg = imread(os.path.join(splitDir, splitImgName))
wholeImg = imread(os.path.join(wholeDir, wholeImgName))

plt.subplot(121)
plt.imshow(splitImg)
plt.subplot(122)
plt.imshow(wholeImg, cmap='gray')

# %%