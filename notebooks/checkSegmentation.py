# %%
from pathlib import Path
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import cv2

from src.visualization.segmentationVis import viewPredictorResult
from src.models import modelTools
from src.data.imageProcessing import imSplit

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
# %%
def preprocess(input_image, magnification_downsample_factor=1.0): 
    #internal variables
    #   median_radius_raw = used in the background illumination pattern estimation. 
    #       this radius should be larger than the radius of a single cell
    #   target_median = 128 -- LIVECell phase contrast images all center around a 128 intensity
    median_radius_raw = 75
    target_median = 128.0
    
    #large median filter kernel size is dependent on resize factor, and must also be odd
    median_radius = round(median_radius_raw*magnification_downsample_factor)
    if median_radius%2==0:
        median_radius=median_radius+1

    #scale so mean median image intensity is 128
    input_median = np.median(input_image)
    intensity_scale = target_median/input_median
    output_image = input_image.astype('float')*intensity_scale

    #define dimensions of downsampled image image
    dims = input_image.shape
    y = int(dims[0]*magnification_downsample_factor)
    x = int(dims[1]*magnification_downsample_factor)

    #apply resizing image to account for different magnifications
    output_image = cv2.resize(output_image, (x,y), interpolation = cv2.INTER_AREA)
    
    #clip here to regular 0-255 range to avoid any odd median filter results
    output_image[output_image > 255] = 255
    output_image[output_image < 0] = 0

    #estimate background illumination pattern using the large median filter
    background = cv2.medianBlur(output_image.astype('uint8'), median_radius)
    output_image = output_image.astype('float')/background.astype('float')*target_median

    #clipping for zernike phase halo artifacts
    output_image[output_image > 180] = 180
    output_image[output_image < 70] = 70
    output_image = output_image.astype('uint8')

    return output_image
# %%
# experiment = 'TJ2321-LPD4Lin1'
experiment = 'TJ2302'
# %%
# dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
# datasetDictPath = Path(f'../data/{experiment}/{experiment}DatasetDicts.npy')
# datasetDicts = np.load(datasetDictPath, allow_pickle=True)
dataPath = '../data/TJ2321-LPD4Lin1/raw/phaseContrast'
# dataPath = '../data/TJ2302/split4/phaseContrast'
imPaths = [f'{dataPath}/{imName}' for imName in os.listdir(dataPath)]

# %%
predictorMe = modelTools.getSegmentModel('../models/sartoriusBT474')
# %%
from centermask.config import get_cfg
from detectron2.engine import DefaultPredictor

homePath = '..'
confidenceThresh = 0.6

cfg = get_cfg()
cfg.merge_from_file(f'{homePath}/data/sartorius/configs/bt474_config.yaml')
cfg.MODEL.WEIGHTS = f"{homePath}/models/segmentation/LIVECell_anchor_free_model.pth"  # path to the model we just trained
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidenceThresh
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidenceThresh
cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidenceThresh
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidenceThresh
predictorSartorius = DefaultPredictor(cfg)
# predictorSartorius = modelTools.getLIVECell()

# %%
imgNum = 0
imPath = imPaths[imgNum]
img = imread(imPath)
pcTiles = imSplit(img, nIms = 4)
img = pcTiles[1]

imgProcess = preprocess(img)
imgProcess3 = np.array([imgProcess, imgProcess, imgProcess]).transpose(1, 2, 0)

plt.subplot(121)
plt.imshow(imgProcess3)
plt.subplot(122)
plt.imshow(imgProcess, cmap = 'gray')
# %%
imgNum = 300
imPath = imPaths[imgNum]
img = imread(imPath)
pcTiles = imSplit(img, nIms = 4)
img = pcTiles[0]
imgProcess3 = np.array([img, img, img]).transpose(1, 2, 0)

# viewPredictorResult(predictorMe, imgProcess)
viewPredictorResult(predictorSartorius, imgProcess)

# %%
x = imread()
# %%
# im3 = np.array([im, im, im]).transpose([1, 2, 0])
