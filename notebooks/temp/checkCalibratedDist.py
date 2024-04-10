# %% [markdown]
"""
This is a notebook to explore the extent of 
"""

# %%
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from pathlib import Path
# %%
homePath = Path('../../')

greenDirB = homePath / 'data/TJ2442B/raw/greenCalibrated'
greenDirF = homePath / 'data/TJ2442F/raw/greenCalibrated'



# %%
bVals = []
dirBFiles = list(greenDirB.iterdir())
random.shuffle(dirBFiles)
c = 0
for fileName in tqdm(dirBFiles):
    try:
        img = imread(fileName)
    except:
        print(f'{fileName} is not a tiff')
        continue
    
    imgVals = list(img.ravel())
    minVal = np.min(imgVals)
    maxVal = np.max(imgVals)
    meanVal = np.mean(imgVals)
    medianVal = np.median(imgVals)

    imgVals = [minVal, maxVal, meanVal, medianVal]
    bVals += list(imgVals)
    if c > 1000:
        break
    c += 1
# %%
fVals = []
dirFFiles = list(greenDirF.iterdir())
random.shuffle(dirFFiles)
c = 0
for fileName in tqdm(dirFFiles):
    try:
        img = imread(fileName)
    except:
        print(f'{fileName} is not a tiff')
        continue
    
    imgVals = list(img.ravel())
    minVal = np.min(imgVals)
    maxVal = np.max(imgVals)
    meanVal = np.mean(imgVals)
    medianVal = np.median(imgVals)

    imgVals = [minVal, maxVal, meanVal, medianVal]
    fVals += list(imgVals)
    if c > 1000:
        break
    c += 1
# %%
bVals = np.array(bVals)
fVals = np.array(fVals)
plt.subplot(121)
plt.hist(bVals[bVals<50])
plt.title('Yankeelov')
plt.subplot(122)
plt.hist(fVals[fVals<50])
plt.title('Brock')
plt.suptitle('')