# %% [markdown]
"""
This is a script making a data loader using the outline information 
(not loading individual images)
"""
# %%
from src.data.imageProcessing import bbIncrease, splitName2Whole
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from skimage.io import imread
from skimage.transform import resize

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# %%
class singleCellCrop(Dataset):
    """
    Dataloader class for cropping out a cell from an image

    Attributes
    --------------------
    dataPath: Relative path to load images
    phase: Train/test phase
    seed: Random seed for shuffling
    transforms: Transforms for reducing overfitting
    """

    def __init__(self, datasetDicts, transforms, dataPath, phase, randomSeed = 1234):
        """
        Input: 
        - datasetDicts: Catalogs images and cell segmentations in detectron2 format
        - transforms: Transforms images to tensors, resize, etc.
        - dataPath: Path to grab images
        - phase: Train or testing
        - randomSeed: Seed used to shuffle data

        - segmentations: List of polygons of segmentations from datasetDicts
        - phenotypes: List of phenotypes associated with each segmentation
        - imgNames: List of paths to load image
        - bbs: List of bounding boxes for segmentations
        """
        self.dataPath = dataPath
        self.phase = phase
        self.seed = randomSeed
        self.transforms = transforms
        self.segmentations, self.phenotypes, self.imgNames, self.bbs = self.balance(datasetDicts)
        # Variable parameters for segmentation
        self.maxImgSize = 300
        self.nIncrease = 75
        
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
        print(imgCrop.shape)
        # imgCrop = bbIncrease(poly, bb, imgName, img, self.nIncrease)
        # print(imgCrop.shape)
        # Pad image
        diffRows = int((maxRows - imgCrop.shape[0])/2)
        diffCols = int((maxCols - imgCrop.shape[1])/2)
        pcCrop = F.pad(torch.tensor(imgCrop[:,:,0]), pad=(diffCols, diffCols, diffRows, diffRows)).numpy()
        pcCrop = resize(pcCrop, (maxRows, maxCols))

        pcCrop = np.array([pcCrop, pcCrop, pcCrop]).transpose((1,2,0))
        if self.transforms:
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
        maxAmt = min(cts)

        segmentations, phenotypes, imgNames, bbs = self.shuffleLists([segmentations, phenotypes, imgNames, bbs], self.seed)
        uniqueIdx = []
        for pheno in uniquePheno:
            idx = list(np.where(phenotypes == pheno)[0][0:maxAmt])
            uniqueIdx += idx
        random.seed(self.seed)
        random.shuffle(uniqueIdx)
        
        # Create train/test split
        n = int(0.9*len(uniqueIdx))
        if self.phase == 'train':
            uniqueIdx = uniqueIdx[0:n]
        elif self.phase == 'test':
            uniqueIdx = uniqueIdx[n:]
        else:
            raise ValueError('Phase must be train or test')

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
experiment = 'TJ2201'
datasetDicts = np.load(f'../data/{experiment}/split16/{experiment}DatasetDict.npy', allow_pickle=True)
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        # transforms.Resize(356),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]),
}
# %%
dataPath = f'../data/{experiment}/raw/phaseContrast'

image_datasets = {x: singleCellCrop(datasetDicts, data_transforms[x], dataPath, phase=x) 
                    for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True)
                    for x in ['train', 'test']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%
def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15,5))
    plt.imshow(inp)
    plt.show()

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out)
# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            torch.save(model.state_dict(), )

        print()



    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
# %%
model = models.resnet152(pretrained=True)
nIncrease = image_datasets['train'].nIncrease
modelSaveName = f'../../models/classifySingleCellCrop{nIncrease}Resnet152.pth'
num_ftrs = model.fc.in_features

# Create new layer and assign in to the last layer
# Number of output layers is now 2 for the 2 classes
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Scheduler to update lr 
# Every 7 epochs the learning rate is multiplied by gamma
setp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, setp_lr_scheduler, num_epochs=50)
# %%
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), '../../models/classifySplitResnet152.pth')