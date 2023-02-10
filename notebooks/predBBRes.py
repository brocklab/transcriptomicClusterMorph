# %%
from src.models import testBB

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# %%
# First get training/testing results
homePath = Path('..')
modelNames = ['classifySingleCellCrop-688020', 'classifySingleCellCrop-686756', 'classifySingleCellCrop-688997']
datasetDictPath = '../data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy'
# %%
datasetDicts = np.load(datasetDictPath, allow_pickle=True) 
# %%
modelRes = []
for modelName in modelNames:
    probs, allLabels, scores = testBB.getModelResults(modelName, homePath, datasetDicts)
    modelRes.append(testBB.testResults(probs, allLabels, scores, modelName))
# %%
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for res in modelRes:
    modelDetails = testBB.getModelDetails(Path('../results/classificationTraining') / f'{res.modelName}.out')
    print(modelDetails['nIncrease'])
    auc = res.auc
    plotLabel = f'BB increase {modelDetails["nIncrease"]} px, AUC = {auc:0.2f}'
    plt.plot(res.fpr, res.tpr, label=plotLabel, linewidth=3)
plt.legend(fontsize=12)
plt.title('Phenotype Prediction\nIncreasing Bounding Box')
plt.savefig('../figures/bbIncreaseROC.png', dpi=600)