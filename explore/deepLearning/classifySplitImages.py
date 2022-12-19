# %%
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

from skimage.io import imread
import matplotlib.pyplot as plt
import random

from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
# %%
datasetDicts = np.load(f'./TJ2201DatasetDict.npy', allow_pickle=True)
# %%
class esamSplit(Dataset):
    """
    Dataloader class for pytorch
    """
    def __init__(self, files, phenoWellDict, n, transforms, phase, seed=1234):
        self.imDir = imDir
        self.seed = seed
        self.n = n
        self.phase = phase
        self.phenoWellDict = phenoWellDict
        self.files = files
        self.transforms = transforms


        # Get finalized number of images
        phenoCount = {0:0, 1:0}
        phenoIms = {0: [], 1: []}
        finalIms = []
        phenotypes = []

        for file in self.files:
            well = file.split('_')[1]
            
            if well in phenoWellDict.keys() and phenoCount[phenoWellDict[well]]<n:
                phenoIms[phenoWellDict[well]].append(file)
                phenoCount[phenoWellDict[well]] += 1
                

        esamNeg, esamPos = self.shuffleLists([phenoIms[0], phenoIms[1]])

        for esamNegImg, esamPosImg in zip(esamNeg, esamPos):
            finalIms.append(esamNegImg)
            finalIms.append(esamPosImg)
            phenotypes += [0, 1]
        if phase == 'train':
            n = int(len(finalIms)*0.9)
            finalIms = finalIms[0:n]
            phenotypes = phenotypes[0:n]
        elif phase == 'test':
            n = int(len(finalIms)*0.9)
            finalIms = finalIms[n:]
            phenotypes = phenotypes[n:]
        self.finalIms = finalIms
        self.phenotypes = phenotypes
    def __len__(self):
        return len(self.finalIms)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgName = self.finalIms[idx]
        imgPath = os.path.join(imgName)
        img = imread(imgPath)

        label = self.phenotypes[idx]
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def getLabels(self):
            
        return self.phenotypes

    def shuffleLists(self, l):
        random.seed(self.seed)
        l = list(zip(*l))
        random.shuffle(l)
        return list(zip(*l))
# %%
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

monoPos = ['B2','B3','B4','B5','B6','C2','C3','C4','C5','C6','D2','D3','D4','D5','D6']
monoNeg = ['E2','E3','E4','E5','E6','F2','F3','F4','F5','F6','G2','G3','G4','G5','G6']

phenoWellDict = {}
for well in monoNeg:
    phenoWellDict[well] = 0
for well in monoPos:
    phenoWellDict[well] = 1
imDir = '../../data/TJ2201/phaseContrast'
seed = 1234
n = 4860

path = '../../data/TJ2201/TJ2201Split16/phaseContrast'
files = [img['file_name'].split('/')[-1] for img in datasetDicts if len(img['annotations'])>0]
files = [os.path.join(path, file) for file in files]
image_datasets = {x: esamSplit(files, phenoWellDict, n, data_transforms[x], phase=x, seed=1234) 
                    for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True)
                    for x in ['train', 'test']}

# train_dataset = esamSplit(imDir, phenoWellDict, n, phase='train')
# train_loader = DataLoader(train_dataset, batch_size=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%
def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(30,10))
    plt.imshow(inp)
    plt.show()

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out)
# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

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
            torch.save(model.state_dict(), '../../output/classifySingleCellResnet152.pth')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
# %%
model = models.resnet152(pretrained=True)
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
torch.save(model.state_dict(), '../../output/classifySplitResnet152.pth')