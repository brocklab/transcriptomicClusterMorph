# %%
import sys
sys.path.append('../scripts')
import cellMorphHelper

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import matplotlib

from skimage import measure, img_as_float
from skimage.io import imread
from skimage.measure import regionprops
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

        date = '_'.join(self.pcImg.split('_')[3:5])
        self.date = cellMorphHelper.convertDate(date)
        self.well = self.pcImg.split('_')[1]

        masks, pred_classes, actualClassifications, scores = self.assignPhenotype(predictor)

        assert len(pred_classes) == len(actualClassifications)
        self.masks = masks
        self.predClasses = pred_classes
        self.actualClasses = actualClassifications
        self.scores = scores
    def assignPhenotype(self, predictor):
        """
        Compares predicted and output phenotype assignments. Stores mask as well. 
        """
        imgPc = imread(self.pcImg)
        imgComp = imread(self.compositeImg)
        outputs = predictor(imgPc)['instances']        
        masks = outputs.pred_masks.numpy()
        pred_classes = outputs.pred_classes.numpy()
        scores = outputs.scores.numpy()
        actualClassifications, fluorescentPredClasses, classScores, finalMasks = [], [], [], []
        # For each mask, compare its identity to the predicted class
        for mask, pred_class, score in zip(masks, pred_classes, scores):
            actualColor = findFluorescenceColor(imgComp.copy(), mask.copy())
            if actualColor == 'red':
                classification = 0
            elif actualColor == 'green':
                classification = 1
            else:
                continue
            actualClassifications.append(classification)
            fluorescentPredClasses.append(pred_class)
            classScores.append(score)
            finalMasks.append(mask)
        return (finalMasks, fluorescentPredClasses, actualClassifications, scores)

    def imshow(self):
        """Temp function to show image, TODO: Add labels"""
        imgPc = imread(self.pcImg)
        plt.imshow(imgPc)
# %%
writeData = 0

if writeData:
    print('Grabbing classifications')
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
else:
    imgLog=pickle.load(open('../data/fullClassificationAccuracy.pickle',"rb"))
# %%
matplotlib.rcParams.update({'font.size': 12})
dateAccuracy = {'E2': {}, 'D2': {}, 'E7': {}}
for img in imgLog:
    date = '_'.join(img.pcImg.split('_')[3:5])
    date = cellMorphHelper.convertDate(date)
    well = img.well
    if date not in dateAccuracy[well].keys():
        dateAccuracy[well][date] = [0, 0]
    
    dateAccuracy[well][date][0] += np.sum(np.array(img.actualClasses) == np.array(img.predClasses))
    dateAccuracy[well][date][1] += len(img.actualClasses)

wells = dateAccuracy.keys()
wellColors = {'E2': 'red', 'D2': 'green', 'E7': 'magenta'}
plt.figure(figsize=(10,5))
for well in wells:
    dates, accuracies = [], []
    for date in dateAccuracy[well].keys():
        dates.append(date)
        accuracies.append(dateAccuracy[well][date][0]/dateAccuracy[well][date][1])
    plt.scatter(dates, accuracies, c=wellColors[well], label=well)

plt.legend()
plt.xlabel('Date')
plt.ylabel('Accuracy')

# %% Filter out low eccentricity cells
minEcc = 0.85
imgLogEccentric = []
pred, actual = [], []
for img in imgLog:
    for i, mask in enumerate(img.masks):
        region = regionprops(mask.astype(np.uint8))
        region = region[0]
        if region.eccentricity>minEcc:
            pred.append(img.predClasses[i])
            actual.append(img.actualClasses[i])

percentAccuracy = np.sum(np.array(pred) == np.array(actual))/len(actual)
print(f'Accuracy: {percentAccuracy:0.3f}')
# %%
total = 0
correct = 0
for img in imgLog:
    if img.well == 'E7':
        total += len(img.actualClasses)
        if len(img.actualClasses)>0:
            correct += np.sum(np.array(img.actualClasses) == np.array(img.predClasses))
print(correct/total)

# %%
total = 0
for img in x:
    total += len(img['annotations'])