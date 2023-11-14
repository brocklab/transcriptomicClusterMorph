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
experiment = 'TJ2321-LPD4Lin1'
modelPath = '../models/sartoriusBT474'
# %%
numClasses = 1
modelPath = Path(modelPath)
if modelPath.parts[-2] != 'segmentation':
    modelPathParts = list(modelPath.parts)
    modelPathParts.insert(-1, 'segmentation')
    modelPath = Path(*modelPathParts)
modelPath = str(modelPath)
cfg = get_cfg()
if not torch.cuda.is_available():
    print('CUDA not available, resorting to CPU')
cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cellMorph_Train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = numClasses
cfg.OUTPUT_DIR = modelPath
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
# Inference
cfg.DETECTION_MAX_INSTANCES = 1000
cfg.POST_NMS_ROIS_INFERENCE = 8000
predictor = DefaultPredictor(cfg)
# %%
compositeImPath = Path(f'../data/{experiment}/raw/composite')
pcPath = Path(f'../data/{experiment}/raw/phaseContrast')
# %%
# from centermask.config import get_cfg
# from detectron2.engine import DefaultPredictor

# confidenceThresh = 0.3
# cfg = get_cfg()
# cfg.merge_from_file('../data/sartorius/configs/bt474_config.yaml')
# cfg.MODEL.WEIGHTS = "../models/segmentation/LIVECell_anchor_free_model.pth"  # path to the model we just trained
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidenceThresh
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidenceThresh
# cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidenceThresh
# cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidenceThresh


# predictor = DefaultPredictor(cfg)
# %%
def getRecord(pcName, compositeImg, pcImg, idx, predictor = predictor):
    phenoCounts = {0: 0, 1: 0}
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
        color = findFluorescenceColor(compositeImg, mask)
        if color == 'red':
            pheno = 1
        else:
            pheno = 0
        phenoCounts[pheno] += 1
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
    return record, phenoCounts
# %%
datasetDictsPath = f'../data/{experiment}/{experiment}DatasetDicts-1.npy'
if Path(datasetDictsPath).exists():
    print('Loading dataset dict')
    datasetDicts = np.load(Path(datasetDictsPath), allow_pickle=True)
    image_ids = [record['image_id'] for record in datasetDicts]
    idx = max(image_ids)
else:
    datasetDicts = []
    idx = 0

# %%
from src.data.fileManagement import convertDate, getModelDetails
import datetime
allPcImsOld = list(pcPath.iterdir())
allPcIms = []
allDates = []
colsExclude = ['10', '11']
for imName in tqdm(allPcImsOld):
    imName = str(imName)
    well = imName.split('_')[1]
    column = well[1:]
    date = '_'.join(imName.split('_')[3:5])
    date = convertDate(date)
    allDates.append(date)
    if date >=  datetime.datetime(2023, 11, 8, 1, 54) and well not in colsExclude:
        compositeName = Path(imName.replace('phaseContrast', 'composite'))
        if compositeName.exists():
            allPcIms.append(Path(imName))
print(len(allPcIms))
allDates = set(allDates)
# %%
datasetDicts = list(datasetDicts)
modIdx = 1
allPhenoCounts = {0: 0, 1: 0}
maxAmt = 150000
print(f'Searching for minimum {maxAmt} cells / phenotype ----')
for pcName in tqdm(allPcIms[idx:]):
    compositeName = Path(str(pcName).replace('phaseContrast', 'composite'))

    pcImg = imread(pcName)
    if compositeName.exists():
        compositeImg = imread(compositeName)
    else:
        compositeImg = np.array([pcImg, pcImg, pcImg]).transpose([1,2,0])

    pcTiles = imSplit(pcImg, nIms = 4)
    compositeTiles = imSplit(compositeImg, nIms = 4)
    
    imNum = 1
    for compositeSplit, pcSplit in zip(compositeTiles, pcTiles):
        pcSplit = np.array([pcSplit, pcSplit, pcSplit]).transpose([1,2,0])
        compositeSplit = compositeSplit[:,:,0:3]
        
        
        newPcImgName = f'{str(pcName)[0:-4]}_{imNum}.png'
        record, phenoCounts = getRecord(pcName = newPcImgName, 
                           compositeImg = compositeSplit, 
                           pcImg = pcSplit, 
                           idx = idx, 
                           predictor = predictor)
        imNum += 1
        idx += 1

        imSplitPath = Path(f'../data/{experiment}/split4/phaseContrast') / Path(newPcImgName).parts[-1]
        imsave(imSplitPath, pcSplit, check_contrast=False)
        datasetDicts.append(record)
        
        if idx % 100 == 0:
            print('Saving...')
            if modIdx % 2 == 0:
                np.save(f'../data/{experiment}/{experiment}DatasetDicts-0.npy', datasetDicts)
                modIdx += 1
            else:
                np.save(f'../data/{experiment}/{experiment}DatasetDicts-1.npy', datasetDicts)
                modIdx += 1
        allPhenoCounts[0] += phenoCounts[0]
        allPhenoCounts[1] += phenoCounts[1]

    if allPhenoCounts[0] > maxAmt and allPhenoCounts[1] > maxAmt:
        break
    
# %%
np.save(f'../data/{experiment}/{experiment}DatasetDicts.npy', datasetDicts)
# %%
