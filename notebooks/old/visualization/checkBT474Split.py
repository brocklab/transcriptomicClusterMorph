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
modelPath = homePath / 'models/segmentation/sartoriusBT474SplitPreprocessed'
predictor = modelTools.getSegmentModel(modelPath)
# %%
dataDir = homePath / 'data/TJ2321-LPD4Lin1/split4/phaseContrast'
imgFiles = list(dataDir.iterdir())
imgFull = imread(imgFiles[300])
splitImgs = imSplit(imgFull, nIms = 4)
img = splitImgs[0]
img = preprocess(img)

plt.imshow(img, cmap = 'gray')
viewPredictorResult(predictor, img)
# %%
sartoriusFiles = list(Path('/home/user/work/cellMorph/data/sartoriusSplit/processedImages/livecell_train_val_images').iterdir())
# sartoriusFiles = list(Path('/home/user/work/cellMorph/data/sartoriusSplit/processedImages/livecell_test_images').iterdir())
imgSart = imread(sartoriusFiles[10])
imgSart = np.array([imgSart, imgSart, imgSart]).transpose([1, 2, 0])
plt.imshow(imgSart)
viewPredictorResult(predictor, imgSart)
# %%
x20 = list(Path('/home/user/work/cellMorph/data/TJ2201/split16/phaseContrast').iterdir())
plt.imshow(imread(x20[480]))
# %%
