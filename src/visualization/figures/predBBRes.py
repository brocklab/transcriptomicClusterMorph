# %%
from src.models import testBB
from src.visualization.trainTestRes import plotTrainingRes
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# %%
# First get training/testing results
homePath = Path('../../../')
modelNames = ['classifySingleCellCrop-713279']
datasetDictPath = homePath / 'data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy'
# %%
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]

# %%
modelRes = []
for modelName in modelNames:
    probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts)
    modelRes.append(testBB.testResults(probs, allLabels, scores, imgNames, modelName))
# %%
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for res in modelRes:
    modelDetails = testBB.getModelDetails(homePath / 'results/classificationTraining' / f'{res.modelName}.out')
    print(modelDetails['nIncrease'])
    auc = res.auc
    plotLabel = f'BB increase {modelDetails["nIncrease"]} px, AUC = {auc:0.2f}'
    plt.plot(res.fpr, res.tpr, label=plotLabel, linewidth=3)
plt.legend(fontsize=12)
plt.title('Phenotype Prediction\nIncreasing Bounding Box')
# plt.savefig('../figures/bbIncreaseROC.png', dpi=600)
# %% Monoculture wells only
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
# %%
modelNames = ['classifySingleCellCrop-713279']
modelRes = []
for modelName in modelNames:
    probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts)
    modelRes.append(testBB.testResults(probs, allLabels, scores, imgNames, modelName))
# %%
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for res in modelRes:
    modelDetails = testBB.getModelDetails(homePath / 'results/classificationTraining' / f'{res.modelName}.out')
    print(modelDetails['nIncrease'])
    auc = res.auc
    plotLabel = f'AUC = {auc:0.2f}'
    plt.plot(res.fpr, res.tpr, label=plotLabel, linewidth=3)
plt.legend(fontsize=12, loc='lower right')
plt.title('Coculture Wells ROC')
plt.savefig(homePath / 'figures/temp/cocultureWellsInit.png', dpi=600)
# %%
modelPath = homePath / 'results' / 'classificationTraining' / 'classifySingleCellCrop-713279.out'
x = plotTrainingRes(modelPath, title = 'Coculture Well Training')
# %% Get results by identity
monoPos = ['B2','B3','B4','B5','B6','C2','C3','C4','C5','C6','D2','D3','D4','D5','D6']
monoNeg = ['E2','E3','E4','E5','E6','F2','F3','F4','F5','F6','G2','G3','G4','G5','G6']
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']


idxMonoPos, idxMonoNeg, idxCoPos, idxCoNeg = [], [], [], []
modelRes = testBB.testResults(probs, allLabels, scores, imgNames, modelName)
idx = 0
for imgName, phenotype in zip(modelRes.imgNames, modelRes.labels):
    well = imgName.split('_')[1]
    if well in monoPos:
        idxMonoPos.append(idx)
    if well in monoNeg:
        idxMonoNeg.append(idx)
    if well in co:
        if phenotype == 0:
            idxCoPos.append(idx)
        else:
            idxCoNeg.append(idx)
    idx += 1

# %%
for idx in [idxMonoPos, idxMonoNeg, idxCoPos, idxCoNeg]:
    preds = modelRes.preds[idx]
    labels = modelRes.labels[idx]
    accuracy = np.sum(preds == labels)/len(labels)
    print(f'Accuracy: {accuracy:0.2}\n')