# %%
from src.models.trainBB import singleCellLoader, getTFModel
from src.data.fileManagement import convertDate, splitName2Whole, collateModelParameters
from src.models import testBB
from src.visualization.trainTestRes import plotTrainingRes
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle 
from tqdm import tqdm
import pandas as pd
from scipy import stats

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torch.nn.functional as F
# %%
# First get training/testing results
homePath = Path('../../../')
modelPath = homePath / 'models' / 'classification'
modelNames = list(modelPath.iterdir())
modelNames = [str(modelName.parts[-1]).split('.')[0] for modelName in modelNames]
modelNames.sort()
datasetDictPathFull = homePath / 'data/TJ2201/split16/TJ2201DatasetDictNoBorderFull.npy'
datasetDictPathPartial = homePath / 'data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy'

# %%
datasetDicts = np.load(datasetDictPathFull, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
# %%
nCells = {0:0, 1:0}
for record in tqdm(datasetDicts):
    for cell in record['annotations']:
        nCells[cell['category_id']] += 1
print(nCells)
# %% Increasing bounding box results
# Load model results, test on reserve well if not already run
resultsFile = homePath / 'results' / 'classificationResults' / 'modelResultsCoCulture.pickle'
if resultsFile.exists():
    modelRes = pickle.load(open(resultsFile, "rb"))
else:
    modelRes = {}

modelNames = [
            #   'classifySingleCellCrop-714689',
            #   'classifySingleCellCrop-713279', 
            #   'classifySingleCellCrop-709125',
              'classifySingleCellCrop-1707264894', # 65 px 
              'classifySingleCellCrop-1707668614', # 25 px
              'classifySingleCellCrop-1707714016', # 00 px
              'classifySingleCellCrop-1709261519', # 55 px
              'classifySingleCellCrop-1709418455', # 15 px
              'classifySingleCellCrop-1709372973', # 45 px
              'classifySingleCellCrop-1709327523'  # 35 px
             ]
for modelName in modelNames:
    if modelName not in modelRes.keys():
        print(modelName)
        probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts)
        modelRes[modelName] = testBB.testResults(probs, allLabels, scores, imgNames, modelName)

pickle.dump(modelRes, open(resultsFile, "wb"))
# %%
# Basic ROC plotting
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for model in modelNames:
    modelPath = homePath / 'results' / 'classificationTraining' / f'{model}.out'
    if not modelPath.exists():
        modelPath = homePath / 'results' / 'classificationTraining' / f'{model}.txt'
    assert modelPath.exists()
    modelDetails = testBB.getModelDetails(modelPath)
    res = modelRes[model]
    auc = res.auc
    plotLabel = f'BB increase {modelDetails["nIncrease"]} px, AUC = {auc:0.2f}'
    plt.plot(res.fpr, res.tpr, label=plotLabel, linewidth=3)

plt.legend(fontsize=12)
plt.title('Subpopulation Prediction\nIncreasing Bounding Box')

# plt.savefig(homePath / 'figures/publication/results/subPopulationBoundingBox.png', dpi = 500, bbox_inches = 'tight')
# %% Instead, plot AUC over nIncrease

increases, aucs = [], []

for model in modelNames:
    modelPath = homePath / 'results' / 'classificationTraining' / f'{model}.out'
    if not modelPath.exists():
        modelPath = homePath / 'results' / 'classificationTraining' / f'{model}.txt'
    assert modelPath.exists()
    modelDetails = testBB.getModelDetails(modelPath)
    print(f'{modelDetails["nIncrease"]}, {modelDetails["maxImgSize"]}')
    res = modelRes[model]
    auc = res.auc
    increases.append(modelDetails['nIncrease'])
    aucs.append(auc)

# plt.plot(increases, aucs)
iA = list(zip(increases, aucs))
iA.sort()
increases = [i for i, a in iA]
aucs = [a for i, a in iA]
plt.figure()
plt.figure(figsize=(8,5))
plt.scatter(increases, aucs, s = 100)
plt.plot(increases, aucs)
plt.xticks(increases)
plt.xlabel('Pixel Increase')
plt.ylabel('AUC')
plt.savefig('../../../figures/publication/results/increasingBBSubpop.png', dpi = 500, bbox_inches = 'tight')
# %% Augmentation results
modelDict = {   
            'No Augmentation': 'classifySingleCellCrop-713279',
            'No Surrounding' : 'classifySingleCellCrop-1693286401',
            'No Texture'     : 'classifySingleCellCrop-1693256617'
}

for augName, modelName in modelDict.items():
    if modelName not in modelRes.keys():
        print(modelName)
        probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts)
        modelRes[modelName] = testBB.testResults(probs, allLabels, scores, imgNames, modelName)
pickle.dump(modelRes, open(resultsFile, "wb"))

# %%
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for augName, modelName in modelDict.items():
    res = modelRes[modelName]
    auc = res.auc
    plotLabel = f'{augName} AUC = {auc:0.2f}'
    plt.plot(res.fpr, res.tpr, label=plotLabel, linewidth=3)


plt.title('Subpopulation Prediction\nImage Augmentation')
plt.legend(fontsize=12, loc='lower right')
plt.savefig(homePath / 'figures/publication/results/subPopulationAugmentation.png', dpi = 500, bbox_inches = 'tight')
# %%