# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.draw import polygon2mask
from pathlib import Path
from tqdm import tqdm
import pickle 

from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode
import detectron2

from src.models.trainBB import makeImageDatasets
from src.models import testBB
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset

from src.models import trainBB
from src.data.fileManagement import getModelDetails
from src.data.imageProcessing import imSplit, findBrightGreen, segmentGreenHigh
# %%
def getGreenRecord(datasetDicts, datasetDictsGreen = []):
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []

        for annotation in record['annotations']:
            annotation['bbox'] = detectron2.structures.BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            # if annotation['category_id'] == 1:
            #     newAnnotations.append(annotation)
        # if len(newAnnotations) > 0:
        #     record['annotations'] = newAnnotations
        #     datasetDictsGreen.append(record)
    return datasetDicts
# for record in tqdm(datasetDictsTreat):
#     for cell in record['annotations']:
#         cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
#         cell['bbox_mode'] = BoxMode.XYXY_ABS
# %%
datasetDictsGreen = {}

datasetDicts = load_coco_json('../data/TJ2342A/TJ2342ASegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2342A'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442B/TJ2442BSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442B'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442C/TJ2442CSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442C'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442D/TJ2442DSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442D'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442E/TJ2442ESegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442E'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442F/TJ2442FSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442F'] = getGreenRecord(datasetDicts, [])
# %%
experiment = 'TJ2442E'
datasetDicts = datasetDictsGreen[experiment]
# %%
homePath = Path('../')
imgDir = homePath / f'data/{experiment}/raw/phaseContrast'
compositeSplit = []

nGreenVals = []
for record in tqdm(datasetDicts):
    for annotation in record['annotations']:
        if annotation['category_id'] != 1:
            continue
        seg = annotation['segmentation'][0]
        polyx = seg[::2]
        polyy = seg[1::2]
        polygonSki = list(zip(polyy, polyx))

        img = homePath / record['file_name']
        imgName = str(img.name)
        imgFull = imgName.split('_')
        imgNum = int(imgFull[-1].split('.png')[0])
        imgFull = '_'.join(imgFull[0:-1]) + '.png'
        # phaseContrastFull = imread(imgDir / imgFull)

        compositeDir = Path(str(imgDir).replace('phaseContrast', 'composite'))
        imgFullComposite = str(imgFull).replace('phaseContrast', 'composite')
        uncalDir = Path(str(imgDir).replace('phaseContrast', 'greenUncalibrated'))
        imgUncal = str(imgFull).replace('phaseContrast', 'greenUncalibrated')
        
        uncalFullDir = uncalDir / imgUncal
        if not uncalFullDir.exists():
            continue
        uncalFull = imread(uncalFullDir)
        compositeFull = imread(compositeDir / imgFullComposite)
        compositeSplit = imSplit(compositeFull, nIms = 4)
        uncalSplit = imSplit(uncalFull, nIms = 4)
        uncalImg = uncalSplit[imgNum - 1]
        compositeImg = compositeSplit[imgNum - 1][:,:,0:3]

        mask = polygon2mask(compositeImg.shape[0:2], polygonSki)
        nGreen = findBrightGreen(compositeImg, mask, thresh = 0)

        nGreenVals.append(nGreen)
        break
    if len(nGreenVals) > 0:
        break
# %%
thresh = 300
minVal = np.min(uncalFull)
uncalFilt = uncalFull.copy()
uncalFilt[uncalFilt < thresh] = minVal

plt.imshow(uncalFilt)
# %%
_, BW = segmentGreenHigh(compositeImg)
plt.imshow(BW)
plt.plot(polyx, polyy)