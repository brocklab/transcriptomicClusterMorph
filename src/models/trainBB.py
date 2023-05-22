from src.data.imageProcessing import bbIncrease, bbIncreaseBlackout
from src.data.fileManagement import splitName2Whole

import random
import numpy as np
import copy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pickle 
from pathlib import Path
from collections import OrderedDict

from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim

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

    def __init__(self, datasetDicts, experiment, transforms, dataPath, nIncrease, phase, maxAmt = 0, randomSeed = 1234):
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
        self.maxAmt = maxAmt
        self.segmentations, self.phenotypes, self.imgNames, self.bbs = self.balance(datasetDicts)
        self.experiment = experiment

        self.nIncrease = nIncrease

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
        imgCrop = bbIncrease(poly, bb, imgName, img, self.nIms, self.nIncrease)

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
        # Split off well for training/testing
        testWell = ['B7']

        if self.phase == 'train':
            datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] not in testWell]
        elif self.phase == 'test':
            datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in testWell]

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

def makeImageDatasets(datasetDicts, dataPath, modelInputs, data_transforms = [], phase = ['train', 'test'], isShuffle=True):
    """
    Creates pytorch image datasets using transforms

    Inputs:
    - datasetDicts: Segmentation information
    - dataPath: Location of images
    - modelInputs: 
    """
    nIncrease    = modelInputs['nIncrease']
    maxAmt       = modelInputs['maxAmt']
    batch_size   = modelInputs['batch_size']
    experiment   = modelInputs['experiment']

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

    image_datasets = {x: singleCellLoader(datasetDicts, experiment, data_transforms[x], dataPath, nIncrease, maxAmt = maxAmt, phase=x) 
                    for x in phase}
    dataset_sizes = {x: len(image_datasets[x]) for x in phase}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=isShuffle)
                        for x in phase}
    
    if len(phase) == 1:
        dataloaders = dataloaders[phase[0]]
        dataset_sizes = dataset_sizes[phase[0]]

    return dataloaders, dataset_sizes

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, savePath, resultsSaveName, num_epochs = 25, best_acc = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Sending to device')
    model.to(device)
    print(model)
    best_model_wts = copy.deepcopy(model.state_dict())
    # if device.type != 'cuda':
    #     raise Exception('Incorrect device')
    for epoch in tqdm(range(num_epochs), leave=False):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], position=0, leave=True):
                # I have no idea why you have to do this but...
                # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-float-for-argument-2-weight/38961/9
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            currentResults = '{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc)
            print(currentResults)

            # deep copy the model
            improvementResults = ''
            if phase == 'test' and epoch_acc > best_acc:
                improvementResults = 'Improved epoch accuracy, updating weights'
                print(improvementResults)
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            with open(resultsSaveName, 'a') as file:
                file.write(f'{currentResults} \n {improvementResults} \n')
        # Always save model on each epoch
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), savePath)

        print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def getTFModel(modelType, modelPath = '', nClassesNew = 2):
    """
    Loads a model and modifies the last portion for transfer learning
    Inputs:
        - modelType: Type of model desired
        - modelPath: Path to load the state dict if provided
        - nClassesNew: New number of classes
    Outputs:
        - model: Pytorch model
    """
    availableModels = ['resnet152', 'vgg16']
    if modelType not in availableModels:
        print("Model not found, resorting to resnet")
        modelType = 'resnet152'
        
    if modelType == 'resnet152':
        model = models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, nClassesNew)

    elif modelType == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[6].in_features
        # Remove last layer
        features = list(model.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, nClassesNew)])
        model.classifier = nn.Sequential(*features)

    if len(str(modelPath)) > 0:
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        modelSaved = torch.load(modelPath, map_location=device)
        modelSavedCorrect = OrderedDict()
        for k, v in modelSaved.items():
            modelSavedCorrect[k.replace('module.', '')] = v
        
        model.load_state_dict(modelSavedCorrect)
        
        model.eval()

    return model
