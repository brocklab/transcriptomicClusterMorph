# %%
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
# %%
def showImgFilt(calImg, minVal = 0, thresh = 5.25, plot = True):
    calFilt = calImg.copy()

    calFilt[calFilt < thresh] = minVal
    # calFilt[calFilt >= thresh] = 1


    if plot:
        plt.figure(figsize = (20, 10))
        plt.subplot(121)
        plt.imshow(compositeImg)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(calFilt)
        plt.axis('off')

    return calFilt
# %%
homePath = Path('../../')
imgNames = homePath / 'data/TJ2442F/raw/greenUncalibrated'
imgNames = list(imgNames.iterdir())
calFilts = []
for imgName in imgNames:
    uncalName = str(imgName).replace('greenUncalibrated', 'greenCalibrated').replace('png', 'tif')
    compositeName = str(imgName).replace('greenCalibrated', 'composite').replace('tif', 'png')

    calImg = imread(imgName)
    uncalImg = imread(uncalName)
    compositeImg = imread(compositeName)

    calFilt = showImgFilt(calImg, thresh = 7)
    calFilts.append(calFilt)
# %%
imgName = imgNames[-3]
uncalName = imgName
calName = str(imgName).replace('greenUncalibrated', 'greenCalibrated').replace('.png', '.tif')
compositeName = str(imgName).replace('greenCalibrated', 'composite').replace('tif', 'png')
calImg = imread(calName)
uncalImg = imread(uncalName)
compositeImg = imread(compositeName)

calFilt = showImgFilt(calImg, thresh=7, plot = False)
plt.figure(figsize=(5,10))
plt.subplot(211)
plt.imshow(calImg)
plt.title('Calibrated Fluorescence')
plt.axis('off')
# plt.subplot(312)
# plt.imshow(compositeImg)
# plt.axis('off')
plt.subplot(212)
plt.imshow(calFilt)
plt.axis('off')
plt.title('Filtered Calibrated Fluorescence')

plt.tight_layout()
# %%
plt.subplot(121)
plt.hist(uncalImg.ravel())
plt.title('Uncalibrated Fluorescence')
plt.subplot(122)
plt.hist(calImg.ravel())
plt.title('Calibrated Fluorescence')
# %%
plt.subplot(131)
plt.imshow(uncalImg)
plt.axis('off')
plt.subplot(132)
plt.imshow(calImg)
plt.axis('off')
plt.subplot(133)
plt.imshow(compositeImg)
plt.axis('off')
# %%
imgName = imgNames[-1]
uncalName = str(imgName).replace('greenCalibrated', 'greenUncalibrated').replace('tif', 'png')
compositeName = str(imgName).replace('greenCalibrated', 'composite').replace('tif', 'png')

calImg = imread(imgName)
uncalImg = imread(uncalName)
compositeImg = imread(compositeName)

plt.imshow(calImg)

calFilt = showImgFilt(calImg, plot = True)
# %%
# from skimage import color, morphology

# footprint = morphology.disk(5)
# res = morphology.white_tophat(calFilt, footprint)
# plt.imshow(res)
# %%