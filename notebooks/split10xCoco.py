# %%
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
import detectron2
import detectron2.data.datasets as datasets
from detectron2.structures import BoxMode

import cv2
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import random
import torch
import os
import numpy as np

from src.data.imageProcessing import imSplit
# %%

# %%
image_root = ''
datasetDictsTrainVal = datasets.load_coco_json(json_file='../data/sartorius/segmentations/train.json', image_root=image_root)
for record in tqdm(datasetDictsTrainVal):
    for cell in record['annotations']:
        cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
        cell['bbox_mode'] = BoxMode.XYXY_ABS

datasetDictsTest = datasets.load_coco_json(json_file='../data/sartorius/segmentations/test.json', image_root=image_root)
for record in tqdm(datasetDictsTest):
    for cell in record['annotations']:
        cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
        cell['bbox_mode'] = BoxMode.XYXY_ABS
# %%
homePath = Path('../data/sartorius/images/livecell_train_val_images')
idx = 31
imgData = datasetDictsTrainVal[idx]
img = imread(homePath / imgData['file_name'])
annotations = imgData['annotations']
plt.imshow(img, cmap = 'gray')
h, w = img.shape
halfHeight  = h//2
halfWidth   = w//2
img1, img2, img3, img4 = imSplit(img, nIms = 4)
anno1,anno2,anno3,anno4 = [], [], [], []

plt.subplot(221)
plt.title('1')
plt.imshow(img1, cmap = 'gray')

plt.subplot(223)
plt.title('2')
plt.imshow(img2, cmap = 'gray')

plt.subplot(222)
plt.title('3')
plt.imshow(img3, cmap = 'gray')

plt.subplot(224)
plt.title('4')
plt.imshow(img, cmap = 'gray')

plt.figure()
plt.imshow(img, cmap = 'gray')
plt.plot(halfWidth, halfHeight+100, 'rx')
plt.axhline(halfHeight)
plt.axvline(halfWidth)
# %%
plt.figure()
plt.imshow(img, cmap = 'gray')
# plt.plot(halfWidth, halfHeight+100, 'rx')
plt.axhline(halfHeight)
plt.axvline(halfWidth)
anno1,anno2,anno3,anno4 = [], [], [], []


# %%
from skimage.draw import polygon
from skimage.segmentation import clear_border
from skimage.morphology import binary_dilation

plt.imshow(img3, cmap = 'gray')
h, w = img3.shape
annoNoBorder = []
for annotation in anno3:
    mask = np.zeros([h, w])
    x = annotation['segmentation'][0][::2].copy()
    y = annotation['segmentation'][0][1::2].copy()
    # plt.plot(x, y)
    poly = np.array([[y,x] for x,y in zip(x, y)])
    rr, cc = polygon(poly[:, 0], poly[:, 1], mask.shape)

    mask[rr, cc] = [1]
    mask = binary_dilation(mask)

    maskCleared = clear_border(mask)
    if np.sum(maskCleared) != 0:
        annoNoBorder.append(annotation)

for annotation in annoNoBorder:
    x = annotation['segmentation'][0][::2].copy()
    y = annotation['segmentation'][0][1::2].copy()
    plt.plot(x, y, c = 'red')

# %% Put it all together

# 1. Split images and assign annotations to split image
# 2. Get rid of annotations that are on border
# 3. Finish dataset dict
# 4. Save image
# %% Functions
imageInfo = datasetDictsTrainVal[30]
def splitImageAnnotations(imageInfo, homePath = '', plot = False):
    """
    Splits COCO annotations from detectron2 datasetdict format. 

    Inputs:
    imageInfo - Complete segmentation information for one image
    homePath - Path for image loading
    plot - Boolean for plotting annotations

    Outputs:
    List of annotations for image split like so:
    -----------------
    |   1   |    3  |
    |       |       |
    |-------|-------|
    |   2   |    4  |
    |       |       |
    -----------------
    """
    homePath = Path(homePath)
    h, w = [imageInfo['height'], imageInfo['width']]
    halfHeight  = h//2
    halfWidth   = w//2

    if plot == True:
        img = imread(homePath / imageInfo['file_name'])
        plt.imshow(img, cmap = 'gray')
    anno1,anno2,anno3,anno4 = [], [], [], []
    annotations = imageInfo['annotations']

    for annotation in annotations:

        annotationOrig = annotation.copy()
        annotation = annotation.copy()


        annotation['segmentation'] = np.array(annotation['segmentation'])

        x1, y1, x2, y2 = annotation['bbox'].copy()
        bbox = annotation['bbox']
        if x1 < halfWidth and y1 < halfHeight:
            annotation['segmentation'] = annotation['segmentation'].tolist()
            c = 'red'
            anno1.append(annotation)
        elif x1 > halfWidth and y1 < halfHeight:
            x1 -= halfWidth
            x2 -= halfWidth
            annotation['segmentation'][0][::2] -= halfWidth
            annotation['bbox'] = [x1, y1, x2, y2]
            annotation['segmentation'] = annotation['segmentation'].tolist()
            anno3.append(annotation)

            c = 'green'
        elif x1 < halfWidth and y1 > halfHeight:
            y1 -= halfHeight
            y2 -= halfHeight
            annotation['segmentation'][0][1::2] -= halfHeight
            annotation['bbox'] = [x1, y1, x2, y2]
            annotation['segmentation'] = annotation['segmentation'].tolist()
            anno2.append(annotation)

            c = 'blue'
        elif x1 > halfWidth and y1 > halfHeight:
            x1 -= halfWidth
            x2 -= halfWidth
            annotation['segmentation'][0][::2] -= halfWidth
            y1 -= halfHeight
            y2 -= halfHeight
            annotation['segmentation'][0][1::2] -= halfHeight
            annotation['bbox'] = [x1, y1, x2, y2]
            annotation['segmentation'] = annotation['segmentation'].tolist()

            c = 'magenta'
            anno4.append(annotation)
        if plot == True:
            x = annotationOrig['segmentation'][0][::2].copy()
            y = annotationOrig['segmentation'][0][1::2].copy()
            plt.plot(x, y, c)
    return [anno1, anno2, anno3, anno4]

def clearSegmentationBorder(anno):
    """
    Clears segmentations that are on the border of an image of fixed size

    Inputs:
    anno - Annotation from datasetDict detectron2 format

    Outputs:
    annoNoBorder - Same format with border segmentations removed
    """
    h, w = (260, 352)
    annoNoBorder = []
    for annotation in anno:
        mask = np.zeros([h, w])
        x = annotation['segmentation'][0][::2].copy()
        y = annotation['segmentation'][0][1::2].copy()
        poly = np.array([[y,x] for x,y in zip(x, y)])
        rr, cc = polygon(poly[:, 0], poly[:, 1], mask.shape)

        mask[rr, cc] = [1]
        mask = binary_dilation(mask)

        maskCleared = clear_border(mask)
        if np.sum(maskCleared) != 0:
            annoNoBorder.append(annotation)
# %%
homePath = Path('../data/sartorius/images/livecell_train_val_images')
newSavePath = Path('../data/sartoriusSplit/images/livecell_train_val_images')
datasetDictsTrainValSplit = []
imgId = 1
for imageInfo in tqdm(datasetDictsTrainVal):
    img = imread(homePath / imageInfo['file_name'])
    splitImgs = imSplit(img, nIms = 4)
    splitAnnos = splitImageAnnotations(imageInfo, homePath, plot = False)

    c = 1
    for img, anno in zip(splitImgs, splitAnnos):
        fileParts = imageInfo['file_name'].split('.tif')[0].split('_')
        fileParts[-1] = f'{fileParts[-1]}-{c}'
        fileName = f"{'_'.join(fileParts)}.png"
        splitImageInfo = {}
        h, w = img.shape
        splitImageInfo['file_name'] = fileName
        splitImageInfo['annotations'] = anno
        splitImageInfo['height'] = h
        splitImageInfo['width'] = w
        splitImageInfo['image_id'] = imgId

        imgId += 1
        c += 1

        filePath = newSavePath / fileName

        imsave(filePath, img)

    datasetDictsTrainValSplit.append(splitImageInfo)
# %%
homePath = Path('../data/sartorius/images/livecell_test_images')
newSavePath = Path('../data/sartoriusSplit/images/livecell_test_images')
datasetDictsTestSplit = []
imgId = 1
for imageInfo in tqdm(datasetDictsTest):
    img = imread(homePath / imageInfo['file_name'])
    splitImgs = imSplit(img, nIms = 4)
    splitAnnos = splitImageAnnotations(imageInfo, homePath, plot = False)

    c = 1
    for img, anno in zip(splitImgs, splitAnnos):
        fileParts = imageInfo['file_name'].split('.tif')[0].split('_')
        fileParts[-1] = f'{fileParts[-1]}-{c}'
        fileName = f"{'_'.join(fileParts)}.png"
        splitImageInfo = {}
        h, w = img.shape
        splitImageInfo['file_name'] = fileName
        splitImageInfo['annotations'] = anno
        splitImageInfo['height'] = h
        splitImageInfo['width'] = w
        splitImageInfo['image_id'] = imgId

        imgId += 1
        c += 1

        filePath = newSavePath / fileName

        imsave(filePath, img)

    datasetDictsTestSplit.append(splitImageInfo)
# %%
datasetName = 'bt474_train'
def getCells(datasetDict):
    return datasetDict

inputs = [datasetDictsTrainValSplit]
if datasetName in DatasetCatalog:
    DatasetCatalog.remove(datasetName)
    MetadataCatalog.remove(datasetName)

DatasetCatalog.register(datasetName, lambda x=inputs: getCells(inputs[0]))
MetadataCatalog.get(datasetName).set(thing_classes=["cell"])

datasets.convert_to_coco_json(datasetName, output_file='../data/sartoriusSplit/segmentations/train.json', allow_cached = False)
# %%
datasetName = 'bt474_test'
def getCells(datasetDict):
    return datasetDict

inputs = [datasetDictsTestSplit]
if datasetName in DatasetCatalog:
    DatasetCatalog.remove(datasetName)
    MetadataCatalog.remove(datasetName)

DatasetCatalog.register(datasetName, lambda x=inputs: getCells(inputs[0]))
MetadataCatalog.get(datasetName).set(thing_classes=["cell"])

datasets.convert_to_coco_json(datasetName, output_file='../data/sartoriusSplit/segmentations/test.json', allow_cached = False)
# %%