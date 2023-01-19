# %%
import sys
sys.path.append('../scripts')
import cellMorphHelper
from cellMorph import imgSegment
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
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
# %%
writeData = 1

if writeData:
    print('Getting segmentations')
    predictor = cellMorphHelper.getSegmentModel('../models/TJ2201Split16')

    imgLog = []
    if 'cellMorph_test' not in DatasetCatalog:
        register_coco_instances("cellMorph_test", {}, '../data/cocoFiles/TJ2201Split16FullWellTest.json', "")
        images = DatasetCatalog['cellMorph_test']()

    c = 0
    # For each image, catalog model outputs and their predicted/actual classes
    for image in tqdm(images):
        imgLog.append(imgSegment(image['file_name'], predictor, modelType='segment'))
        if c % 10 == 0:
            pickle.dump(imgLog, open('../data/fullSegmentationTJ2201Test.pickle', "wb"))

        c += 1
else:
    imgLog=pickle.load(open('../data/fullSegmentationTJ2201Test.pickle',"rb"))
# %%
