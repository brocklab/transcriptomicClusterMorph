# %%
import sys
sys.path.append('../scripts')
import cellMorphHelper

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle 

from skimage import measure, img_as_float
from skimage.io import imread

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
# %%
def findFluorescenceColor(RGB, mask):
    """
    Finds the fluorescence of a cell
    Input: RGB image location
    Output: Color
    """
    # RGB = imread(RGBLocation)
    mask = mask.astype('bool')
    RGB[~np.dstack((mask,mask,mask))] = 0
    nGreen, BW = cellMorphHelper.segmentGreen(RGB)
    nRed, BW = cellMorphHelper.segmentRed(RGB)
    if nGreen>=(nRed+100):
        return "green"
    elif nRed>=(nGreen+100):
        return "red"
    else:
        return "NaN"

class imgSegment:
    """
    Stores information about cell output from image
    """
    def __init__(self, fileName, predictor):
        assert 'phaseContrast' in fileName
        self.pcImg = fileName
        self.compositeImg = self.pcImg.replace('phaseContrast', 'composite')

        self.well = self.pcImg.split('_')[1]

        masks, pred_classes, actualClassifications = self.assignPhenotype(predictor)

        self.masks = masks
        self.pred_classes = pred_classes
        self.actualClassifications = actualClassifications

    def assignPhenotype(self, predictor):
        """
        Compares predicted and output phenotype assignments. Stores mask as well. 
        """
        imgPc = imread(self.pcImg)
        imgComp = imread(self.compositeImg)
        outputs = predictor(imgPc)['instances']
        masks = outputs.pred_masks.numpy()
        pred_classes = outputs.pred_classes.numpy()
        actualClassifications, fluorescentPredClasses = [], []
        # For each mask, compare its identity to the predicted class
        for mask, pred_class in zip(masks, pred_classes):
            actualColor = findFluorescenceColor(imgComp.copy(), mask.copy())
            if actualColor == 'red':
                classification = 0
            elif actualColor == 'green':
                classification = 1
            else:
                continue
            actualClassifications.append(classification)
            fluorescentPredClasses.append(pred_class)
        return (masks, pred_classes, actualClassifications)

    def imshow(self):
        """Temp function to show image, TODO: Add labels"""
        imgPc = imread(self.pcImg)
        plt.imshow(imgPc)
# %%
writeData = 1

if writeData:
    predictor = cellMorphHelper.getSegmentModel('../output/TJ2201Split16ClassifyFull', numClasses=2)

    imgLog = []
    if 'cellMorph_test' not in DatasetCatalog:
        register_coco_instances("cellMorph_test", {}, '../data/cocoFiles/TJ2201Split16FullWellTest.json', "")
        images = DatasetCatalog['cellMorph_test']()

    c = 0
    # For each image, catalog model outputs and their predicted/actual classes
    for image in tqdm(images):
        imgLog.append(imgSegment(image['file_name'], predictor))
        if c % 10 == 0:
            pickle.dump(imgLog, open('../data/fullClassificationAccuracy.pickle', "wb"))

        c += 1