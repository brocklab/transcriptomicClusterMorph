# %%
from pathlib import Path
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import cv2

from src.visualization.segmentationVis import viewPredictorResult
from src.models import modelTools
from src.data.imageProcessing import imSplit, preprocess

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
# %%
homePath = Path('../../../')
dataDir = homePath / 'data/TJ2321-LPD4Lin1/split4/phaseContrast'
imgFiles = list(dataDir.iterdir())
imgFull = imread(imgFiles[0])
splitImgs = imSplit(imgFull, nIms = 4)
img = splitImgs[2]
img = preprocess(img)

plt.imshow(img[:,:,0])
# %%
modelPath = homePath / 'models/segmentation/sartoriusBT474SplitPreprocessed'
predictor = modelTools.getSegmentModel(modelPath)
viewPredictorResult(predictor, img)
plt.figure()
viewPredictorResult(predictor, imgFull)
# %%
