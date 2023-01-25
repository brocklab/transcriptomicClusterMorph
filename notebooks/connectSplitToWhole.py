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
def splitName2Whole(imgName: str):
    """
    Strips split number from image name

    Inputs:
        - imgName: Name of image in format:
            imagingType_Well_IncucyteNum_Date_Time_ImgNum.extension
    Outputs:
        - imgNameWhole: Name of image in format:
            imagingType_Well_IncucyteNum_Date_Timeß.extension

    """
    extSplit = imgName.split('.')
    ext = extSplit[-1]
    imgName = extSplit[0]

    imgNameWhole = '_'.join(imgName.split('_')[0:-1])+'.'+ext
    return imgNameWhole

def split2WholeCoords(nIms, wholeImgSize):
    """
    Returns coordinates to connect split images to whole images

    Inputs:
        - nIms: This is the number of images an original image was split into
        - wholeImgSize: 1x2 list of type [nRows, nCols]

    Outputs:
        - coordinates: A dictionary where keys are the split number and 

    Example:
    coordinates = split2WholeCoords(nIms=16, wholeImg=img)
    # polyx and polyy are the initial segmentation coordinates
    polyxWhole = polyx + coordinates[int(splitNum)][0]
    polyyWhole = polyy + coordinates[int(splitNum)][1]
    """ 

    div = int(np.sqrt(nIms))
    nRow = wholeImgSize[0]
    nCol = wholeImgSize[1]

    M = nRow//div
    N = nCol//div
    tiles = []
    imNum = 1
    coordinates = {}
    for x in range(0,wholeImgSize[1],N): # Column
        for y in range(0,wholeImgSize[0],M): # Row
            coordinates[imNum] = [x, y]
            imNum += 1

    return coordinates

def expandImageSegmentation(poly, bb, splitNum, coords):
    """
    Connects split image to whole
    
    """
    poly = np.array(imgSeg['annotations'][0]['segmentation'][0])
    polyx = poly[::2]
    polyy = poly[1::2]

    cIncrease = coords[int(splitNum)]
    bbIncrease = bb.copy()
    bbIncrease[1] += cIncrease[1]
    bbIncrease[3] += cIncrease[1]
    bbIncrease[0] += cIncrease[0]
    bbIncrease[2] += cIncrease[0]

    polyxWhole = polyx + cIncrease[0]
    polyyWhole = polyy + cIncrease[1]
    
    return [polyxWhole, polyyWhole, bbIncrease]
# %%
datasetDicts = np.load('../data/TJ2201/split16/TJ2201DatasetDict.npy', allow_pickle=True)
# %%
# 'phaseContrast_C7_1_2022y04m07d_04h00m_1.png'
splitDir = '../data/TJ2201/split16/phaseContrast'
wholeDir = '../data/TJ2201/raw/phaseContrast'

# Find an image with segmentations
# np.random.seed(1234)
imgNum = np.random.randint(len(datasetDicts))
while True:
    imgSeg = datasetDicts[imgNum]
    splitNum = int(imgSeg['file_name'].split('_')[-1].split('.')[0])
    fileNameSplit = imgSeg['file_name'].split('/')[-1]
    fileNameWhole = splitName2Whole(fileNameSplit)
    if len(imgSeg['annotations']) == 0:
        imgNum += 1
    else:
        break

imgSplit = imread(os.path.join(splitDir, fileNameSplit))
imgWhole = imread(os.path.join(wholeDir, fileNameWhole))

coords = split2WholeCoords(nIms = 16, wholeImgSize = imgWhole.shape)
poly = np.array(imgSeg['annotations'][0]['segmentation'][0])
bb = imgSeg['annotations'][0]['bbox']
bb = np.array([int(corner) for corner in bb])
polyx = poly[::2]
polyy = poly[1::2]
polyxWhole, polyyWhole, bbIncrease = expandImageSegmentation(poly, bb, splitNum, coords)

imgBB = imgSplit[bb[1]:bb[3], bb[0]:bb[2]]
imgBBWhole = imgWhole[bbIncrease[1]:bbIncrease[3], bbIncrease[0]:bbIncrease[2]]
plt.subplot(131)
plt.imshow(imgSplit)
plt.plot(polyx, polyy, c='red')
plt.subplot(132)
plt.imshow(imgBB)
plt.subplot(133)
plt.imshow(imgBBWhole, cmap='gray')


# %%
