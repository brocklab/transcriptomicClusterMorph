# %% [markdown]
"""
This is a notebook meant to test the network's ability to classify cells without context, as I suspect it cannot do it without knowing there are other cells of similar type but I am not sure.
"""
# %%
import sys
sys.path.append('../scripts')
import cellMorphHelper

import torch
import detectron2

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random, pickle
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

# Import image processing
from skimage import measure
from skimage import img_as_float
from skimage.io import imread
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border
from scipy.spatial import ConvexHull

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
# %% Get Predictor
from detectron2.engine import DefaultTrainer
cfg = get_cfg()
if not torch.cuda.is_available():
    print('CUDA not available, resorting to CPU')
    cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cellMorph_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (cell). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = '../output/TJ2201Split16Classify'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
predictor = DefaultPredictor(cfg)
# %% Get images/known segmentations

# Set information for desired experiment/well
experiment = 'TJ2201'
stage = 'test'
wells = ['D2', 'E2']
# Will store cropped images
ims = {well: [] for well in wells}
# Grab segmentations
segDir = os.path.join('../data',experiment,'segmentedIms')
segFiles = os.listdir(segDir)
segFiles = [segFile for segFile in segFiles if segFile.endswith('.npy') and segFile.split('_')[1] in wells]
# Get training data
random.seed(1234)
random.shuffle(segFiles)
trainPercent = 0.75
trainNum = int(trainPercent*len(segFiles))
if stage == 'train':
    segFiles = segFiles[0:trainNum]
elif stage == 'test':
    segFiles = segFiles[trainNum:]

# Loop through segmentation files
for segFile in tqdm(segFiles):
    well = segFile.split('_')[1]
    segFull = os.path.join(segDir, segFile)
    seg = np.load(segFull, allow_pickle=True)
    seg = seg.item()

    splitMasks = cellMorphHelper.imSplit(seg['masks'])
    nSplits = len(splitMasks)

    splitDir = f'{experiment}Split{nSplits}'
    imgBase = cellMorphHelper.getImageBase(seg['filename'].split('/')[-1])    
    # Segmentation files are for full images, so this splits them in the same way
    for splitNum in range(1, len(splitMasks)+1):
        imgFile = f'phaseContrast_{imgBase}_{splitNum}.png'
        imgFileComposite = f'composite_{imgBase}_{splitNum}.png'

        imgPath = os.path.join('../data', splitDir, 'phaseContrast', imgFile)
        compositePath = os.path.join('../data', splitDir, 'composite', imgFileComposite)
        assert os.path.isfile(imgPath)

        img = imread(imgPath)
        mask = splitMasks[splitNum-1]
        cellNums = np.unique(mask)
        cellNums = cellNums[cellNums != 0]

        # Each cell is assigned a specific number
        for cellNum in cellNums:
            contours = measure.find_contours(img_as_float(mask==cellNum), .5)
            fullContour = np.vstack(contours)

            px = fullContour[:,1]
            py = fullContour[:,0]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            bb = [np.min(px), np.min(py), np.max(px), np.max(py)]
            bb = [int(x) for x in bb]
            imgBlank = img.copy()
            isCell = np.where(mask!=cellNum)
            imgBlank[isCell[0], isCell[1]] = 128

            imgCrop = imgBlank[bb[1]:bb[3], bb[0]:bb[2]].copy() 
            ims[well].append(imgCrop)
# %% Put together example images
random.seed(1234)
randomD2 = random.sample(range(0, len(ims['D2'])), 4)
randomE2 = random.sample(range(0, len(ims['E2'])), 4)

whichWell = 0
imgCrops = []
for cellNum in range(4):
    imgCrops.append(ims['D2'][randomD2[cellNum]])
    imgCrops.append(ims['E2'][randomE2[cellNum]])

imgCrop = ims['D2'][0]
imgBlank = np.zeros(img.shape)+128.0
imgBlank = imgBlank.astype(np.uint8)
row = 0
col = 150
imgBlank[row:row+imgCrop.shape[0], col:col+imgCrop.shape[1]] = imgCrop

plt.imshow(imgBlank)

# %% Testing 
fileName = 'phaseContrast_E2_4_2022y04m08d_08h00m_1.png'
imgPath = os.path.join('../data', splitDir, 'phaseContrast', fileName)
img = imread(imgPath)
# %%
imgName = 'testMono2.png'
# cellMorphHelper.viewPredictorResult(predictor, './testMono.png')
outputs = predictor(imread(imgName))