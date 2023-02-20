from src.data.imageProcessing import bbIncrease
from src.data.fileManagement import splitName2Whole
from src.models import trainBB

import random
import numpy as np
import copy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pickle 
from pathlib import Path

from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim

def predictDataset(datasetDict, model):
    pass

def testModel(model, loaders, testSummaryPath='') -> list:
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    probs = []
    allLabels = []
    scores = []
    running_corrects = 0
    for inputs, labels in tqdm(loaders['test'], position=0, leave=True):
        # I have no idea why you have to do this but...
        # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-float-for-argument-2-weight/38961/9
        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels.to(device)


        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        probs.append(outputs.cpu().data.numpy())
        allLabels.append(labels.cpu().data.numpy())
        scores.append(F.softmax(outputs, dim=1).cpu().data.numpy())
        running_corrects += torch.sum(preds == labels.data)
        

    probs = np.concatenate(probs)
    allLabels = np.concatenate(allLabels)
    scores = np.concatenate(scores)

    modelTestResults = {'probs': probs, 'allLabels': allLabels, 'scores': scores}
    if len(testSummaryPath)>0:
        pickle.dump(modelTestResults, open(testSummaryPath, "wb"))

    return [probs, allLabels, scores]

def getModelDetails(outPath) -> dict:
    """
    Splits .out file from slurm and returns dictionary with details for recreating the model

    Inputs:
        - outPath: Location of .out file
    Outputs:
        - modelDetails: Dictionary of model parameters
    """
    with open(outPath) as outFile:
        x = outFile.read()
    detailsSplit = x.split('~ Model Details ~')[1]
    modelDetailsOut = detailsSplit.split('-'*10)[0]
    modelDetailsOut = modelDetailsOut.split('\n')[1:-1]
    modelDetails = {}
    for detail in modelDetailsOut:
        detail, val = detail.split(' - ')
        if val.isdigit():
            val = int(val)
        modelDetails[detail] = val

    return modelDetails

def getModelResults(modelName, homePath, datasetDicts, modelType = 'resnet152'):
    """
    Gets results of a model on testing data

    Inputs:
        - modelName: Used to identify the model name/output file
        - homePath: Path to return to home directory
        - datasetDicts: Segmentation information in detectron2 format
        - modelType: Type of model (used for transfer learning)
    
    Outputs:
        - probs: Returned probabilities of class identification from network
        - allLabels: True labels
        - scores: softmax of probs to get actual probabilities summed to 1

    """
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')

    # TODO: Could turn this into a function if > 1 models are used
    if modelType == 'resnet152':
        model = models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(modelPath, map_location=device))
        model.eval()

        num_ftrs = model.fc.in_features

        # Create new layer and assign in to the last layer
        # Number of output layers is now 2 for the 2 classes
        # model.fc = nn.Linear(num_ftrs, 2)

        model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()

    outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')

    modelDetails = getModelDetails(outPath)

    dataPath = Path.joinpath(homePath, 'data', modelDetails['experiment'], 'raw', 'phaseContrast')

    dataloaders, dataset_sizes = trainBB.makeImageDatasets(datasetDicts,
                                                dataPath, 
                                                nIncrease    = modelDetails['nIncrease'], 
                                                maxAmt       = modelDetails['maxAmt'], 
                                                batch_size   = modelDetails['batch_size'],
                                                isShuffle = False
                                                )
    probs, allLabels, scores = testModel(model, dataloaders)
    imgNames = dataloaders['test'].dataset.imgNames
    return [probs, allLabels, scores, imgNames]

class testResults:
    """
    A class for holding testing results from a pytorch model prediction
    Attributes
    ----
    - name: The name of the report
    - probs: Output "probabilites" from pytorch model
    - scores: The softmax of the probabilities
    - labels: The actual labels    
    """
    def __init__(self, probs, allLabels, scores, imageNames, modelName, resultName = ''):
        self.probs =  probs
        self.labels = allLabels
        self.scores = scores
        self.name = resultName
        self.n = self.probs.shape[0]
        self.preds = np.argmax(self.scores, axis= 1)
        self.imgNames = imageNames
        if resultName == '':
            self.name = modelName
        self.modelName = modelName

        self.fpr, self.tpr, self.auc = self.getROC()
        self.acc = np.sum(self.labels == self.preds)/self.n
        
        # modelPath = Path.joinpath('..')
    def getROC(self):
        fpr, tpr, _ = roc_curve(self.labels, self.scores[:,1])
        roc_auc = roc_auc_score(self.labels, self.scores[:,1])
        return fpr, tpr, roc_auc
    
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.__str__()
