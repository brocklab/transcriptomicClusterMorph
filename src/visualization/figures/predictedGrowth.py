# %% [markdown]
"""
Residual plot of predicted vs actual growth
"""
# %%
from src.models.trainBB import singleCellLoader
from src.models import testBB
from src.data.fileManagement import convertDate, splitName2Whole

from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import matplotlib.dates as mdates
from scipy import stats

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
# %%
homePath = Path('../../../')
datasetDictPath = homePath / './data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy'
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
# %%
wells = []
wells = [seg['file_name'].split('_')[1] for seg in datasetDicts]
uniqueWells, wellCts = np.unique(wells, return_counts=True)
# %%
modelName = 'classifySingleCellCrop-688020'
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(modelPath, map_location=device))
model.eval()

num_ftrs = model.fc.in_features

# Create new layer and assign in to the last layer
# Number of output layers is now 2 for the 2 classes
model.fc = nn.Linear(num_ftrs, 2)


model.load_state_dict(torch.load(modelPath, map_location=device))
model.eval()

outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')

modelDetails = testBB.getModelDetails(outPath)

# %%
experiment  = 'TJ2201'
nIncrease   = 65
maxAmt      = 15000
batch_size  = 40
num_epochs  = 40
modelType   = 'resnet152'

dataPath = Path(homePath / f'data/{experiment}/raw/phaseContrast')

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
image_datasetTrain = singleCellLoader(datasetDicts, 
                                data_transforms['train'], 
                                dataPath, 
                                nIncrease = modelDetails['nIncrease'], 
                                phase='train', 
                                maxAmt = modelDetails['maxAmt'])
trainImgs = list(set(image_datasetTrain.imgNames))
wellsSave = ['E2', 'D2', 'E7']
datasetDictsSub = []
for seg in tqdm(datasetDicts):
    imgName = Path(seg['file_name']).parts[-1]
    well = seg['file_name'].split('_')[1]
    if imgName not in trainImgs and well in wellsSave:
        datasetDictsSub.append(seg)

image_dataset = singleCellLoader(datasetDictsSub, 
                                data_transforms['train'], 
                                dataPath, 
                                nIncrease = modelDetails['nIncrease'], 
                                phase='none', 
                                maxAmt = None)
dataset_sizes = len(image_dataset)
dataloader = DataLoader(image_dataset, batch_size=modelDetails['batch_size'], shuffle=False)


# %%
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
probs = []
allLabels = []
scores = []
running_corrects = 0
for inputs, labels in tqdm(dataloader, position=0, leave=True):
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

modelTestResults = {'probs': probs, 'allLabels': allLabels, 'scores': scores, 'images': dataloader.dataset.imgNames}
pickle.dump(modelTestResults, open(homePath / 'results/classificationResults/TJ2201PredGrowth.pickle', "wb"))
# %%
modelTestResults = pickle.load(open(homePath / 'results/classificationResults/TJ2201PredGrowth.pickle',"rb"))

# %%
fpr, tpr, _ = roc_curve(modelTestResults['allLabels'], modelTestResults['scores'][:,1])
roc_auc = roc_auc_score(modelTestResults['allLabels'], modelTestResults['scores'][:,1])
print(f'AUC: {roc_auc:0.3}')
# %% Get ground truth and prediction results
groundTruth = {}
for seg in datasetDictsSub:
    fileName = splitName2Whole(Path(seg['file_name']).parts[-1])
    if fileName not in groundTruth.keys():
        groundTruth[fileName] = [0, 0]
    for annotation in seg['annotations']:
        category_id = annotation['category_id']
        groundTruth[fileName][category_id] += 1

predictedLabels = np.argmax(modelTestResults['scores'], axis=1)
predicted = {}
for predLabel, imgName in zip(predictedLabels, modelTestResults['images']):
    imgName = splitName2Whole(imgName)
    if imgName not in predicted.keys():
        predicted[imgName] = [0, 0]
    predicted[imgName][predLabel] += 1
# %%
dfPred = pd.DataFrame(predicted).transpose()
dfTrue = pd.DataFrame(groundTruth).transpose()

dfFull = dfPred.join(dfTrue, lsuffix='_pred', rsuffix='_true')

wells = [img.split('_')[1] for img in dfFull.index]
dates = [convertDate('_'.join(img.split('_')[3:5])) for img in dfFull.index]
dfFull['well'] = wells
dfFull['dates'] = dates
dfFull.head()
# %%
def getWell(dfFull, well):
    """Extracts data for only specified well"""
    dfWell = dfFull[dfFull['well'] == well]
    dfWell = dfWell.sort_values(by=['dates'])
    dfWell = pd.DataFrame(dfWell.groupby('dates').sum()).reset_index()
    return dfWell

def ccc(x,y):
    vx, cov_xy, cov_xy, vy = np.cov(x,y, bias=True).flat
    mx, my = x.mean(), y.mean()
    return 2*cov_xy / (vx + vy + (mx-my)**2)
# %%
dfWell = getWell(dfFull, 'D2')   
plt.scatter(dfWell['0_pred'], dfWell['0_true'],  color='green')

slope, intercept, r_value, p_value, std_err = stats.linregress(dfWell['0_pred'], dfWell['0_true'])

plt.title(f'R^2 = {r_value**2:.3}')
# %%
dfWell = getWell(dfFull, 'E2')   
plt.scatter(dfWell['1_pred'], dfWell['1_true'],  color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(dfWell['1_pred'], dfWell['1_true'])

plt.title(f'R^2 = {r_value**2:.3}')
# %%
dfWell = getWell(dfFull, 'E7')   
plt.scatter(dfWell['0_pred'], dfWell['0_true'],  color='green')
plt.scatter(dfWell['1_pred'], dfWell['1_true'],  color='red')

slope, intercept, r_value0, p_value, std_err = stats.linregress(dfWell['0_pred'], dfWell['0_true'])
slope, intercept, r_value1, p_value, std_err = stats.linregress(dfWell['1_pred'], dfWell['1_true'])

plt.title(f'R^2 = {r_value0**2:.3}')

# %%


image_datasetNew = singleCellLoader(datasetDictsSub, 
                                data_transforms['train'], 
                                dataPath, 
                                nIncrease = modelDetails['nIncrease'], 
                                phase='none', 
                                maxAmt = None)
# %%
trainImgs = list(set(image_datasetTrain.imgNames))
datasetDictsTest = []
for seg in tqdm(datasetDicts):
    if Path(seg['file_name']).parts[-1] not in trainImgs:
        datasetDictsTest.append(seg)

# %%
monoPos = ['B2','B3','B4','B5','B6','C2','C3','C4','C5','C6','D2','D3','D4','D5','D6']
monoNeg = ['E2','E3','E4','E5','E6','F2','F3','F4','F5','F6','G2','G3','G4','G5','G6']
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']


nMonoPos, nMonoNeg, nCoPos, nCoNeg = 0, 0, 0, 0
for phenotype, imgName in zip(image_datasetTrain.phenotypes, image_datasetTrain.imgNames):
    well = imgName.split('_')[1]
    if well in monoPos:
        nMonoPos += 1
    if well in monoNeg:
        nMonoNeg += 1
    if well in co:
        if phenotype == 1:
            nCoPos += 1
        else:
            nCoNeg += 1
        
print(nMonoPos)
print(nMonoNeg)
print(nCoPos)
print(nCoNeg)
