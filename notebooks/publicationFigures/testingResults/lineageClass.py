# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.data.fileManagement import getModelDetails
from src.models import testBB

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle


# %%
homePath = Path('../../../')
experiment = 'TJ2321-LPD4Lin1'
dataPath =          homePath / 'data' / Path(f'{experiment}/split4/phaseContrast')
datasetDictPath =   homePath / 'data' / Path(f'{experiment}/{experiment}DatasetDicts-0.npy')
datasetDicts =      list(np.load(datasetDictPath, allow_pickle=True))
dataPath =          homePath / 'data' / Path(f'{experiment}/raw/phaseContrast')

# %%
wellSize = {}
for seg in datasetDicts:
    well = seg['file_name'].split('_')[1]
    if well not in wellSize.keys():
        wellSize[well] = [0, 0]
    for cell in seg['annotations']:
        catID = int(cell['category_id'])
        wellSize[well][catID] += 1

np.array(list(wellSize.values())).sum(axis = 0)
# %%
resultsFile = homePath / 'results' / 'classificationResults' / 'modelResultsCoCulture.pickle'
if resultsFile.exists():
    modelRes = pickle.load(open(resultsFile, "rb"))
else:
    modelRes = {}

modelNames = [
              'classifySingleCellCrop-1700167069',
              'classifySingleCellCrop-1700187095',
              'classifySingleCellCrop-1700413166'              
             ]
for modelName in modelNames:
    if modelName not in modelRes.keys():
        probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts)
        modelRes[modelName] = testBB.testResults(probs, allLabels, scores, modelName, imgNames)
        print(modelRes[modelName].auc)
pickle.dump(modelRes, open(resultsFile, "wb"))
# %%
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for model in modelNames:
    modelDetails = testBB.getModelDetails(homePath / 'results' / 'classificationTraining' / f'{model}.txt')
    res = modelRes[model]
    auc = res.auc
    plotLabel = f'BB increase {modelDetails["nIncrease"]} px, AUC = {auc:0.2f}'
    plt.plot(res.fpr, res.tpr, label=plotLabel, linewidth=3)

plt.legend(fontsize=12, loc = 'lower right')
plt.title('Lineage Identification')
plt.savefig(homePath / 'figures/publication/results/lineageIdentification.png', dpi = 500, bbox_inches = 'tight')
