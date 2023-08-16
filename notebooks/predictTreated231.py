# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.models import modelTools
from src.data.fileManagement import splitName2Whole
from src.data.imageProcessing import bbIncrease, bbIncreaseBlackout

from pathlib import Path
import numpy as np
import sys
from tqdm import tqdm
import pickle
import os
import random 
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

import detectron2.data.datasets as datasets
from detectron2.data import MetadataCatalog, DatasetCatalog
# %%
experiment      = 'TJ2201 and TJ2301-231C2'
nIncrease       = 25
maxAmt          = 10000
batch_size      = 64
num_epochs      = 32
modelType       = 'resnet152'
optimizer       = 'sgd'
augmentation    = None
notes = 'Initial run'

modelID, idSource = modelTools.getModelID(sys.argv)
modelSaveName = Path(f'../models/classification/classifySingleCellCrop-{modelID}.pth')
resultsSaveName = Path(f'../results/classificationTraining/classifySingleCellCrop-{modelID}.txt')
modelInputs = {

'experiment'    : experiment, 
'nIncrease'     : nIncrease,
'maxAmt'        : maxAmt,
'batch_size'    : batch_size,
'num_epochs'    : num_epochs,
'modelType'     : modelType,
'modelName'     : modelSaveName.parts[-1],
'modelIDSource' : idSource,
'notes'         : notes,
'optimizer'     : optimizer,
'augmentation'  : augmentation

}
# %%
experiment = 'TJ2201'
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
datasetDictsPre = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDictsPre = [seg for seg in datasetDictsPre if seg['file_name'].split('_')[1] in co]
datasetDictsPre = [record for record in datasetDictsPre if len(record['annotations']) > 0]
# %%
datasetDictsTreat = datasets.load_coco_json(json_file='../data/TJ2301-231C2/TJ2301-231C2Segmentations.json', image_root='')
datasetDictsTreat = [record for record in datasetDictsTreat if len(record['annotations']) > 0]
# %%
# Set labels appropriately
for image in tqdm(datasetDictsTreat):
    for cell in image['annotations']:
        cell['category_id'] = 1
for image in tqdm(datasetDictsPre):
    for cell in image['annotations']:
        cell['category_id'] = 0
# %%
class singleCellLoader(Dataset):
    """
    Dataloader class for cropping out a cell from an image

    Attributes
    --------------------
    - dataPath: Relative path to load images
    - experiment: Experiment being trained/tested
    - phase: Train/test phase
    - seed: Random seed for shuffling
    - transforms: Transforms for reducing overfitting
    - segmentations: List of polygons of segmentations from datasetDicts
    - phenotypes: List of phenotypes associated with each segmentation
    - imgNames: List of paths to load image
    - bbs: List of bounding boxes for segmentations
    """
    def __init__(self, datasetDicts, transforms, dataPath, phase, modelInputs, randomSeed = 1234):
        """
        Input: 
        - datasetDicts: Catalogs images and cell segmentations in detectron2 format
        - transforms: Transforms images to tensors, resize, etc.
        - dataPath: Path to grab images
        - nIncrease: Amount to increase bounding box around cell
        - phase: Train or testing
        - randomSeed: Seed used to shuffle data
        """
        self.dataPath = dataPath
        self.phase = phase
        self.seed = randomSeed
        self.transforms = transforms
        self.maxAmt = modelInputs['maxAmt']
        self.segmentations, self.phenotypes, self.imgNames, self.bbs = self.balance(datasetDicts)
        self.experiment = modelInputs['experiment']
        self.nIncrease = modelInputs['nIncrease']
        self.augmentation = modelInputs['augmentation']
        # Static parameters for segmentation
        experimentParamsLoc = dataPath
        c = 0
        while experimentParamsLoc.name != 'data':
            experimentParamsLoc = experimentParamsLoc.parent
            c += 1
            assert c < 1000
                
        experimentParamsLoc = experimentParamsLoc / 'experimentParams.pickle'
        experimentParams = pickle.load(open(experimentParamsLoc,"rb"))
        self.maxImgSize = experimentParams[self.experiment]['maxImgSize']
        self.nIms = experimentParams[self.experiment]['nIms']
        
        
    def __len__(self):
        return len(self.imgNames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        imgName = self.imgNames[idx]
        imgNameWhole = splitName2Whole(imgName)
        label = self.phenotypes[idx]
        fullPath = os.path.join(self.dataPath, imgNameWhole)
        maxRows, maxCols = self.maxImgSize, self.maxImgSize
        img = imread(fullPath)

        bb = self.bbs[idx]
        poly = self.segmentations[idx]
        nIncrease = self.nIncrease
        colMin, rowMin, colMax, rowMax = bb
        rowMin -= nIncrease
        rowMax += nIncrease
        colMin -= nIncrease
        colMax += nIncrease

        # imgOrig = imread(Path('../data/TJ2201/split16/phaseContrast/') / imgName)
        # poly2 = np.array(poly)
        # polyx = poly2[:,0]
        # polyy = poly2[:,1]
        # plt.figure()
        # plt.imshow(imgOrig, cmap = 'gray')
        # plt.plot(polyx, polyy, c = 'red')
        # plt.title(label)

        # Indexing checks
        if rowMin <= 0:
            rowMin = 0
        if rowMax > img.shape[0]:
            rowMax = img.shape[0]
        if colMin <= 0:
            colMin = 0
        if colMax >= img.shape[1]:
            colMax = img.shape[1]

        # Increase the size of the bounding box and crop
        bbIncreased = [colMin, rowMin, colMax, rowMax]
        imgCrop = img[bbIncreased[1]:bbIncreased[3], bbIncreased[0]:bbIncreased[2]]

        # imgCrop = bbIncreaseBlackout(poly, bb, imgName, img, self.nIms, label, self.nIncrease)
        imgCrop = bbIncrease(poly, bb, imgName, img, self.nIms, self.nIncrease, augmentation=self.augmentation)

        # Pad image
        diffRows = int((maxRows - imgCrop.shape[0])/2)
        diffCols = int((maxCols - imgCrop.shape[1])/2)
        pcCrop = F.pad(torch.tensor(imgCrop), pad=(diffCols, diffCols, diffRows, diffRows)).numpy()
        pcCrop = resize(pcCrop, (maxRows, maxCols))

        pcCrop = np.array([pcCrop, pcCrop, pcCrop]).transpose((1,2,0))
        if self.transforms:
            pcCrop = Image.fromarray(np.uint8(pcCrop*255))
            img = self.transforms(pcCrop)
        return img, label

    def plotResults(self, idx):
        img, label = self.__getitem__(idx)
        img = img.numpy().transpose((1, 2, 0))
        plt.imshow(img, cmap='gray')

    def balance(self, datasetDicts):
        """
        Balances the data so that there are equal inputs. 

        Input: 
            - datasetDicts, detectron2 format for segmentations
        Output:
            - segmentation, numpy array of nx2
            - phenotypes, list of encoded phenotypes
            - imgNames, list of image names
        """
        # Reformat dataset dict to most relevant information
        segmentations, phenotypes, imgNames, bbs = [], [], [], []
        # Note there is a lot of repeats for images but this is much cleaner
        for img in datasetDicts:
            imgName = os.path.basename(img['file_name'])
            for annotation in img['annotations']:
                segmentations.append(np.array(annotation['segmentation'][0]))
                phenotypes.append(annotation['category_id'])
                imgNames.append(imgName)
                bbs.append([int(corner) for corner in annotation['bbox']])
        # Balance dataset
        uniquePheno, cts = np.unique(phenotypes, return_counts=True)
        
        if self.maxAmt == 0:
            maxAmt = min(cts)
        else:
            maxAmt = self.maxAmt
            
        if maxAmt > min(cts):
            self.maxAmt = min(cts)

        segmentations, phenotypes, imgNames, bbs = self.shuffleLists([segmentations, phenotypes, imgNames, bbs], self.seed)
        uniqueIdx = []

        if self.phase == 'train':
            for pheno in uniquePheno:
                
                idx = list(np.where(phenotypes == pheno)[0][0:self.maxAmt])
                uniqueIdx += idx
        else:
            uniqueIdx = list(range(0, len(phenotypes)))
        random.seed(self.seed)
        random.shuffle(uniqueIdx)
        
        self.uniqueIdx = uniqueIdx
        # Get finalized amts
        segmentations = np.array([np.reshape(seg, (int(len(seg)/2), 2)) for seg in segmentations[uniqueIdx]], dtype='object')
        phenotypes = phenotypes[uniqueIdx]
        imgNames = imgNames[uniqueIdx]
        bbs = bbs[uniqueIdx]
        return [segmentations, phenotypes, imgNames, bbs]
    
    @staticmethod
    def shuffleLists(l, seed=1234):
        random.seed(seed)
        l = list(zip(*l))
        random.shuffle(l)

        return [np.array(itm, dtype='object') for itm in list(zip(*l))]
