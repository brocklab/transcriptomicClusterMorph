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

datasetDicts = np.load('./TJ2201DatasetDict.npy', allow_pickle=True)
# %%
# 'phaseContrast_C7_1_2022y04m07d_04h00m_1.png'
idx = 403

splitDir = '../../data/TJ2201/split16/phaseContrast'
wholeDir = '../../data/TJ2201/raw/phaseContrast'

imgs = [img['file_name'] for img in datasetDicts]

splitImgName = imgs[idx].split('/')[-1]
wholeImgName = splitName2Whole(splitImgName)
splitImg = imread(os.path.join(splitDir, splitImgName))
wholeImg = imread(os.path.join(wholeDir, wholeImgName))

splitNum = splitImgName.split('.')[0].split('_')[-1]

# plt.figure(figsize=(10,10))
# plt.subplot(121)
# plt.imshow(splitImg)
# plt.title(splitNum)
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(wholeImg, cmap='gray')
# plt.axis('off')

# %% Figure out coordinate calculations
nIms = 16
div = int(np.sqrt(nIms))
nRow = wholeImg.shape[0]
nCol = wholeImg.shape[1]

M = nRow//div
N = nCol//div
tiles = []
imNum = 1
coordinates = {}
for x in range(0,wholeImg.shape[1],N): # Column
    for y in range(0,wholeImg.shape[0],M): # Row
        coordinates[imNum] = [x, y]
        imNum += 1
# %% Get mask coordinates
for imgSeg in datasetDicts:
    imgName = imgSeg['file_name'].split('/')[-1]
    if imgName == splitImgName:
        break
assert imgName == splitImgName

poly = np.array(imgSeg['annotations'][0]['segmentation'][0])
polyx = poly[::2]
polyy = poly[1::2]

polyxWhole = polyx + coordinates[int(splitNum)][0]
polyyWhole = polyy + coordinates[int(splitNum)][1]

plt.figure(figsize=(100,100))
plt.subplot(121)
plt.imshow(splitImg)
plt.plot(polyx, polyy, c='red', linewidth=20)
plt.subplot(122)
plt.imshow(wholeImg, cmap='gray')
plt.plot(polyxWhole, polyyWhole, linewidth=20)
# %% Finalized Functions
def split2WholeCoords(nIms, wholeImg):
    """
    Returns coordinates to connect split images to whole images

    Inputs:
        - nIms: This is the number of images an original image was split into
        - wholeImg: The initial image that was split 

    Outputs:
        - coordinates: A dictionary where keys are the split number and 

    Example:
    coordinates = split2WholeCoords(nIms=16, wholeImg=img)
    # polyx and polyy are the initial segmentation coordinates
    polyxWhole = polyx + coordinates[int(splitNum)][0]
    polyyWhole = polyy + coordinates[int(splitNum)][1]
    """ 

    div = int(np.sqrt(nIms))
    nRow = wholeImg.shape[0]
    nCol = wholeImg.shape[1]

    M = nRow//div
    N = nCol//div
    tiles = []
    imNum = 1
    coordinates = {}
    for x in range(0,wholeImg.shape[1],N): # Column
        for y in range(0,wholeImg.shape[0],M): # Row
            coordinates[imNum] = [x, y]
            imNum += 1

    return coordinates