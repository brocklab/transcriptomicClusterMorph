# %%
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.draw import polygon2mask

from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.data.datasets as datasets

from src.data import imageProcessing
# %%
def convertRecords(datasetDicts):
    newDatasetDicts = []
    for record in tqdm(datasetDicts):
        well = record['file_name'].split('_')[1]
        if well.endswith('10') or well.endswith('11'):
            continue
        record = record.copy()

        for annotation in record['annotations']:
            annotation['bbox'] = BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
        newDatasetDicts.append(record)
    return newDatasetDicts
# %%
experiment  = 'TJ2453-436Co'

datasetDicts = load_coco_json(f'../data/{experiment}/{experiment}Segmentations.json', '.')
datasetDicts = convertRecords(datasetDicts)
# %%
c = 1
plot = 0
for record in tqdm(datasetDicts):
    fileName = record['file_name']
    fileSplit = fileName.split('_')
    imNum = int(fileSplit[-1].split('.png')[0])
    fileName = '_'.join(fileSplit[0:-1])+'.png'
    img = imread(fileName.replace('phaseContrast', 'composite'))
    img = img[:,:,0:-1]
    imgsSplit = imageProcessing.imSplit(img, nIms = 16)
    splitSize = imgsSplit[0].shape
    newAnnotations = []
    for annotation in record['annotations']:
        poly = annotation['segmentation'][0]
        polyx = poly[::2]
        polyy = poly[1::2]
        polygonSki = list(zip(polyy, polyx))

        mask = polygon2mask(splitSize[0:-1], polygonSki)
        mask = mask.astype('int')

        RGB = imgsSplit[imNum-1].copy()
        mask = mask.astype('bool')

        RGB[~np.dstack((mask,mask,mask))] = 0
        nGreen, BW = imageProcessing.segmentGreen(RGB)
        nRed, BW   = imageProcessing.segmentRed(RGB)
        
        # This will remove fusion or misclassified cells
        if nGreen > 150 and nRed > 150:
            continue
        else:
            newAnnotations.append(annotation)
        
    record['annotations'] = newAnnotations
        # Plotting
        # message = f'#Green:{nGreen}\n#Red:{nRed}'
        # t = plt.text(polyx[0], polyy[0], message, fontsize=10)
        # t.set_bbox(dict(facecolor='blue', alpha=0.5, edgecolor='blue'))
        # plt.plot(polyx, polyy, c = 'black', linewidth = 2)
    
# %%
def getCells(datasetDict):
    return datasetDict
inputs = [datasetDicts]
if 'cellMorph' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph')
    MetadataCatalog.remove('cellMorph')

DatasetCatalog.register("cellMorph", lambda x=inputs: getCells(inputs[0]))
MetadataCatalog.get("cellMorph").set(thing_classes=["cell"])

datasets.convert_to_coco_json('cellMorph', output_file=f'../data/{experiment}/{experiment}SegmentationsFiltered.json', allow_cached=False)
