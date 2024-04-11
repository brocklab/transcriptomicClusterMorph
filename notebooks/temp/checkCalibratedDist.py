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
from src.data import imageProcessing
from skimage import morphology, measure
from skimage.draw import polygon2mask

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
# %%
def filterCalibratedFluoro(calImg, compositeImg = [], minVal = 0, thresh = 5.25, plot = False):
    calFilt = calImg.copy()
    # background = restoration.rolling_ball(calFilt)
    # calFilt = calFilt - background
    if thresh == 'adaptive':
        thresh = np.mean(calFilt.ravel()) + 8*MAD(calFilt.ravel())
    calFilt[calFilt < thresh] = minVal
    
    BW = calFilt.copy()
    BW[BW > 0 ] = 1
    BW = morphology.binary_dilation(BW)
    BW = morphology.binary_dilation(BW)

    labels = measure.label(BW)
    props = measure.regionprops(labels)
    for prop in props:
        labelNum = prop.label
        if prop.area > 2500:
            labels[labels == labelNum] = 0
        elif prop.area < 10:
            labels[labels == labelNum] = 0
    BW[labels <= 0] = 0
    calFilt = calFilt * BW

    if plot:
        assert len(compositeImg) > 0
        plt.figure(figsize = (20, 10))
        plt.subplot(121)
        plt.imshow(compositeImg)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(calFilt)
        plt.axis('off')

    return calFilt, props

def MAD(x):
    n = len(x)
    return np.mean(np.abs(x - np.mean(x)))
# %%
greenDirB = homePath / 'data/TJ2442B/raw/composite'
greenDirF = homePath / 'data/TJ2442F/raw/composite'

files = list(greenDirF.iterdir())
random.shuffle(files)
ratio = 3

# files[0] = Path('../../data/TJ2442F/raw/composite/composite_C6_1_2024y02m27d_19h38m.png')
# files[0] = fileAbberation
allProps = []
for i in range(5):
    imgComposite = files[i]
    imgCal = Path(str(imgComposite).replace('composite', 'greenCalibrated').replace('.png', '.tif'))
    assert imgCal.exists()

    imgComposite = imread(imgComposite)
    imgCal = imread(imgCal)
    imgCalRav = imgCal.ravel()
    adaptThresh = np.mean(imgCalRav) + 8*MAD(imgCalRav)
    imgFilt, props =filterCalibratedFluoro(imgCal, thresh = adaptThresh)
    print(adaptThresh)
    plt.figure(figsize = (6*ratio, 3*ratio))
    plt.subplot(131)
    plt.imshow(imgCal)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(imgFilt)
    plt.axis('off')
    plt.subplot(133)
    plt.hist(imgCal.ravel())
    plt.axvline(x = adaptThresh, c = 'red')

    allProps += props
    
# %%
area = []
for prop in allProps:
    area.append(prop.area)
area = np.array(area)
plt.hist(area[area < 100])

# %% Check against segmentations
from detectron2.structures import BoxMode
from detectron2.data.datasets import load_coco_json

def getGreenRecord(datasetDicts):
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []

        for annotation in record['annotations']:
            annotation['bbox'] = BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            # if annotation['category_id'] == 1:
            #     newAnnotations.append(annotation)
        # if len(newAnnotations) > 0:
        #     record['annotations'] = newAnnotations
        #     datasetDictsGreen.append(record)
    return datasetDicts
# %%
datasetDicts = load_coco_json('../../data/TJ2342A/TJ2342ASegmentations.json', '.')
datasetDicts = getGreenRecord(datasetDicts)
# %%
allGreen = []
for idx in tqdm(range(len(datasetDicts))):
    record = datasetDicts[idx]
    fileName = '../' + record['file_name']
    fileSplit = fileName.split('_')
    imNum = int(fileSplit[-1].split('.png')[0])
    fileName = '_'.join(fileSplit[0:-1])+'.png'
    img = imread(fileName)
    imgComposite = imread(fileName.replace('phaseContrast', 'composite'))
    imgCal = imread(fileName.replace('phaseContrast', 'greenCalibrated').replace('.png', '.tif'))
    
    imgsSplit = imageProcessing.imSplit(img, nIms = 16)
    compositesSplit = imageProcessing.imSplit(imgComposite, nIms = 16)
    calsSplit = imageProcessing.imSplit(imgCal, nIms = 16)

    imgSplit = imgsSplit[imNum - 1]
    compositeSplit = compositesSplit[imNum-1]
    calSplit = calsSplit[imNum-1]

    imgFilt, props =filterCalibratedFluoro(imgCal, thresh = 'adaptive')

    # plt.figure()
    newAnnotations = []
    for annotation in record['annotations']:
        poly = annotation['segmentation'][0]
        polyx = poly[::2]
        polyy = poly[1::2]
        polygonSki = list(zip(polyy, polyx))
        mask = polygon2mask(img.shape[0:2], polygonSki)

        maskGreen= mask.astype(int)*imgFilt
        nGreen = np.sum(maskGreen)
        if nGreen > 0:
            allGreen.append(nGreen)
        # plt.plot(polyx, polyy, linewidth = 2)
    # plt.imshow(img, cmap = 'gray')

    if idx > 500:
        break
# %%
