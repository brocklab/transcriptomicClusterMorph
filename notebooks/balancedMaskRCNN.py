# %% [markdown]
"""
This will classify cells using mask rcnn 
"""
# %%
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
import torch
# %%
experiment = 'TJ2201'
# %%
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorder.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
# %%
for seg in tqdm(datasetDicts):

    filePath = Path(seg['file_name'])
    filePath = '../' / Path(*filePath.parts[2:])
    seg['file_name'] = str(filePath)
    assert filePath.exists()
# %% Balance dataset dicts so that there is a roughly equal number of cells

nCellsMax = 15000
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDictTrain = []
totalPhenoCount = {0: 0, 1: 0}
imgPhenoCounts = {}
for idx, seg in enumerate(datasetDicts):
    nCells = len(seg['annotations'])
    well = seg['file_name'].split('_')[1]
    if nCells == 0:
        continue
    if well not in co:
        continue
    cts = [0, 0]
    for cell in seg['annotations']:
        cts[cell['category_id']] += 1
    if cts[0] == cts[1] or cts[0] > cts[1]:
       totalPhenoCount[0] += cts[0]
       totalPhenoCount[1] += cts[1]
       
       datasetDictTrain.append(seg)
for idx, seg in enumerate(datasetDicts):
    nCells = len(seg['annotations'])
    well = seg['file_name'].split('_')[1]
    if nCells == 0:
        continue
    if well not in co:
        continue
    cts = [0, 0]
    for cell in seg['annotations']:
        cts[cell['category_id']] += 1
    if cts[1] > cts[0]:
        totalPhenoCount[0] += cts[0]
        totalPhenoCount[1] += cts[1]
        datasetDictTrain.append(seg)
    if totalPhenoCount[1] > totalPhenoCount[0]:
        break
# %%
random.seed(1234)
random.shuffle(datasetDictTrain)

nTrain = int(.9*len(datasetDictTrain))
ddTrain = datasetDictTrain[0:nTrain]
ddTest = datasetDictTrain[nTrain:]

newPhenoCount = [0, 0]
for seg in ddTrain:
    for cell in seg['annotations']:
        newPhenoCount[cell['category_id']] += 1 
print(newPhenoCount)
# %%
def registerDatasetDict(phase):
 
    random.seed(1234)
    random.shuffle(datasetDictTrain)

    nTrain = int(.9*len(datasetDictTrain))

    ddTrain = datasetDictTrain[0:nTrain]
    ddTest = datasetDictTrain[nTrain:]

    if phase == 'train':
        return ddTrain
    elif phase == 'test':
        return ddTest
    
name = 'cocultureClassifier'
imgType = 'phaseContrast'
trainName = f'{name}_train'
testName = f'{name}_test'
if trainName in DatasetCatalog:
    DatasetCatalog.remove(trainName)
    print('Removing training')
if testName in DatasetCatalog:
    DatasetCatalog.remove(testName)
    print('Removing testing')
inputs = [experiment, imgType, 'train']

DatasetCatalog.register(trainName, lambda x = inputs: registerDatasetDict('train'))
MetadataCatalog.get(trainName).set(thing_classes=["cell"])

DatasetCatalog.register(testName, lambda x=inputs: registerDatasetDict('test'))
MetadataCatalog.get(testName).set(thing_classes=["cell"])

# %%
cfg = get_cfg()
if not torch.cuda.is_available():
    print('CUDA not available, resorting to CPU')
    cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cocultureClassifier_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (cell). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = '../models/segmentation/cocultureClassify'
# %%
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
# %%
