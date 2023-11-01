# %%
from src.visualization.segmentationVis import  viewPredictorResult
from src.data.imageProcessing import imSplit, findFluorescenceColor
from src.models import modelTools

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
import detectron2.data.datasets as datasets

from skimage.io import imread, imsave
from skimage import measure

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import torch
import os
from tqdm import tqdm
# %matplotlib inline
# %%
experiment = 'TJ2303-LPD4'

compositeImPath = Path(f'../data/{experiment}/raw/composite')
pcPath = Path(f'../data/{experiment}/raw/phaseContrast')
predictor = modelTools.getSegmentModel('../models/sartoriusBT474')
# %%
def getRecord(pcName, compositeImg, pcImg, idx, predictor = predictor):
    # Go through each cell in each cropped image
    record = {}
    record['file_name'] = pcName
    record['image_id'] = idx
    record['height'] = pcImg.shape[0]
    record['width'] =  pcImg.shape[1]

    outputs = predictor(pcImg)['instances'].to("cpu")  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    nCells = len(outputs)
    cells = []
    # Get segmentation outlines
    for cellNum in range(nCells):
        mask = outputs[cellNum].pred_masks.numpy()[0]
        pheno = 0

        contours = measure.find_contours(mask, .5)
        if len(contours) < 1:
            continue
        fullContour = np.vstack(contours)

        px = fullContour[:,1]
        py = fullContour[:,0]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]
        
        bbox = list(outputs[cellNum].pred_boxes.tensor.numpy()[0])

        cell = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": pheno,
        }

        cells.append(cell)
    record["annotations"] = cells
    return record
# %%
datasetDictsPath = f'../data/{experiment}/{experiment}Segmentations.json'
if Path(datasetDictsPath).exists():
    print('Loading dataset dict')
    datasetDicts = datasets.load_coco_json(json_file=datasetDictsPath, image_root='')

    image_ids = [record['image_id'] for record in datasetDicts]
    idx = max(image_ids)
else:
    datasetDicts = []
    idx = 0
# %%
datasetDicts = []
idx = 0
datasetDicts = list(datasetDicts)
modIdx = 1
allPcIms = list(pcPath.iterdir())
for pcName in tqdm(allPcIms[idx:]):
    compositeName = Path(str(pcName).replace('phaseContrast', 'composite'))
    pcImg = imread(pcName)
    if compositeName.exists():
        compositeImg = imread(compositeName)
    else:
        compositeImg = np.array([pcImg, pcImg, pcImg]).transpose([1,2,0])

    pcTiles = imSplit(pcImg, nIms = 16)
    
    imNum = 1
    for pcSplit in pcTiles:
        pcSplit = np.array([pcSplit, pcSplit, pcSplit]).transpose([1,2,0])
        compositeSplit = []
        
        
        newPcImgName = f'{str(pcName)[0:-4]}_{imNum}.png'
        record = getRecord(pcName = newPcImgName, 
                           compositeImg = compositeSplit, 
                           pcImg = pcSplit, 
                           idx = idx, 
                           predictor = predictor)
        
        imNum += 1
        idx += 1

        imSplitPath = Path(f'../data/{experiment}/split16/phaseContrast') / Path(newPcImgName).parts[-1]
        imsave(imSplitPath, pcSplit, check_contrast=False)
        datasetDicts.append(record)
        
        if idx % 100 == 0:
            print('Saving...')
            def getCells(datasetDict):
                return datasetDict

            inputs = [datasetDicts]
            if 'cellMorph' in DatasetCatalog:
                DatasetCatalog.remove('cellMorph')
                MetadataCatalog.remove('cellMorph')

            DatasetCatalog.register("cellMorph", lambda x=inputs: getCells(inputs[0]))
            MetadataCatalog.get("cellMorph").set(thing_classes=["cell"])

            datasets.convert_to_coco_json('cellMorph', output_file='./test', allow_cached=False)
        break
def getCells(datasetDict):
    return datasetDict
inputs = [datasetDicts]
if 'cellMorph' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph')
    MetadataCatalog.remove('cellMorph')

DatasetCatalog.register("cellMorph", lambda x=inputs: getCells(inputs[0]))
MetadataCatalog.get("cellMorph").set(thing_classes=["cell"])

datasets.convert_to_coco_json('cellMorph', output_file=datasetDictsPath, allow_cached=False)
np.save(f'../data/{experiment}/{experiment}DatasetDicts.npy', datasetDicts)
# %%imSplit