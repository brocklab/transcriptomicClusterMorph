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
import argparse

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

import detectron2.data.datasets as datasets
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2
from detectron2.structures import BoxMode

# %%
# %% Add argparse
parser = argparse.ArgumentParser(description='Network prediction parameters')
parser.add_argument('--experiment', type = str, metavar='experiment',  help = 'Experiment to run')
parser.add_argument('--nIncrease',  type = int, metavar='nIncrease',   help = 'Increase of bounding box around cell')
parser.add_argument('--maxAmt',     type = int, metavar='maxAmt',      help = 'Max amount of cells')
parser.add_argument('--batch_size', type = int, metavar='batch_size',  help = 'Batch size')
parser.add_argument('--num_epochs', type = int, metavar='num_epochs',  help = 'Number of epochs')
parser.add_argument('--modelType',  type = str, metavar='modelType',   help = 'Type of model (resnet, vgg, etc.)')
parser.add_argument('--notes',      type = str, metavar='notes',       help = 'Notes on why experiment is being run')
parser.add_argument('--optimizer',  type = str, metavar='optimizer',   help = 'Optimizer type')
parser.add_argument('--augmentation',  type = str, metavar='augmentation',   help = 'Image adjustment (None, blackoutCell, stamp)')

# This is for running the notebook directly
args, unknown = parser.parse_known_args()
# %%
experiment      = 'TJ2201 and TJ2301-231C2'
nIncrease       = 25
maxAmt          = 50000
batch_size      = 64
num_epochs      = 32
modelType       = 'resnet152'
optimizer       = 'sgd'
augmentation    = None
nIms            = 16
maxImgSize      = 150
notes = 'Second run for accuracy'

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
'nIms'          : nIms,
'maxImgSize'    : maxImgSize,
'augmentation'  : augmentation

}

argItems = vars(args)

for item, value in argItems.items():
    if value is not None:
        print(f'Replacing {item} value with {value}')
        modelInputs[item] = value
modelDetailsPrint = modelTools.printModelVariables(modelInputs)

# %%
experiment = 'TJ2454-2311kb3'
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
# datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
# datasetDictsPre = np.load(datasetDictPath, allow_pickle=True)
# co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
# datasetDictsPre = [seg for seg in datasetDictsPre if seg['file_name'].split('_')[1] in co]
# datasetDictsPre = [record for record in datasetDictsPre if len(record['annotations']) > 0]

datasetDictsPre = datasets.load_coco_json(json_file='../data/TJ2454-2311kb3/TJ2454-2311kb3Segmentations.json', image_root='')
datasetDictsPre = [record for record in datasetDictsPre if len(record['annotations']) > 0]
for record in tqdm(datasetDictsPre):
    for cell in record['annotations']:
        cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
        cell['bbox_mode'] = BoxMode.XYXY_ABS
# %%
datasetDictsTreat = datasets.load_coco_json(json_file='../data/TJ2301-231C2/TJ2301-231C2SegmentationsNoBorder.json', image_root='')
datasetDictsTreat = [record for record in datasetDictsTreat if len(record['annotations']) > 0]
for record in tqdm(datasetDictsTreat):
    for cell in record['annotations']:
        cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
        cell['bbox_mode'] = BoxMode.XYXY_ABS
# %%
# Set labels appropriately
for image in tqdm(datasetDictsTreat):
    for cell in image['annotations']:
        cell['category_id'] = 1
for image in tqdm(datasetDictsPre):
    for cell in image['annotations']:
        cell['category_id'] = 0

imgPaths = {0: '../data/TJ2454-2311kb3/raw/phaseContrast',
            1: '../data/TJ2301-231C2/raw/phaseContrast'}

modelInputs['imgPaths'] = imgPaths
# %% Replace file paths
for image in datasetDictsPre:
    filePath = Path(image['file_name'])
    # image['file_name'] = str(Path(*filePath.parts[1:]))
for image in datasetDictsTreat:
    filePath = Path(image['file_name'])
    image['file_name'] = str(filePath).replace('raw', 'split16')

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
        self.maxImgSize = modelInputs['maxImgSize']
        self.nIms = modelInputs['nIms']
        self.imgPaths = modelInputs['imgPaths']
        
    def __len__(self):
        return len(self.imgNames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        imgName = self.imgNames[idx]
        imgNameWhole = splitName2Whole(imgName)
        label = self.phenotypes[idx]

        dataPath = self.imgPaths[self.phenotypes[idx]]
        fullPath = os.path.join(dataPath, imgNameWhole)
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
            uniqueIdx = list(range(0, len(phenotypes)))[0:1000]
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

# %%
datasetDicts = datasetDictsPre + datasetDictsTreat
phase = ['train', 'test']
data_transforms = []
batch_size   = modelInputs['batch_size']

mean = np.array([0.4840, 0.4840, 0.4840])
std = np.array([0.1047, 0.1047, 0.1047])
if data_transforms == []:
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            # transforms.Resize(356),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'none': transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    print('Not using custom transforms')

# image_datasets = {x: singleCellLoader(datasetDicts, experiment, data_transforms[x], dataPath, nIncrease, maxAmt = maxAmt, phase=x)
image_datasets = {x: singleCellLoader(datasetDicts, data_transforms[x], dataPath, phase=x, modelInputs = modelInputs) 
                for x in phase}
dataset_sizes = {x: len(image_datasets[x]) for x in phase}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=0)
                    for x in phase}
# %%
inputs, classes = next(iter(dataloaders['train']))
# %%
# for idx in np.where(classes.numpy() == 1)[0]:
#     plt.figure()
#     plt.imshow(inputs[idx].numpy().transpose((1,2,0)))
#     plt.title(classes[idx].numpy())

#     if idx > 25:
#         break
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not modelSaveName.parent.exists():
    raise NotADirectoryError('Model directory not found')

model = getTFModel(modelInputs['modelType'])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
modelDetailsPrint = modelTools.printModelVariables(modelInputs)

with open(resultsSaveName, 'a') as file:
    file.write(modelDetailsPrint)

# Scheduler to update lr
# Every 7 epochs the learning rate is multiplied by gamma
setp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, 
                    criterion, 
                    optimizer, 
                    setp_lr_scheduler, 
                    dataloaders, 
                    dataset_sizes, 
                    modelSaveName,
                    resultsSaveName,
                    num_epochs=modelInputs['num_epochs'],
                    nImprove = 5
                    )
# %%
