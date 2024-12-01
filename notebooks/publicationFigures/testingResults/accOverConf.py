# %%
import pickle
import datetime
from pathlib import Path
import numpy as np

from src.data.fileManagement import convertDate, getModelDetails

# %%
def getModelResults(homePath, resultsFile, resName):

    if resultsFile.exists():
        modelRes = pickle.load(open(resultsFile, "rb"))
    res = modelRes['classifySingleCellCrop-1707668614']
    return res
homePath = Path('../../../')
resultsFile = homePath / 'results' / 'classificationResults' / 'modelResultsCoCulture.pickle'

resName231 = 'classifySingleCellCrop-1707668614'
resName436 = 'classifySingleCellCrop-1715386736'
resNameTreat = ''

# res = getModelResults()
# %%
homePath = Path('../../../')
resultsFile = homePath / 'results' / 'classificationResults' / 'modelResultsCoCulture.pickle'
if resultsFile.exists():
    modelRes = pickle.load(open(resultsFile, "rb"))
# %%
res = modelRes[resName436]
allDates = [convertDate('_'.join(name.split('_')[3:5])) for name in res.name]
startDate, endDate = min(allDates), max(allDates)
idxDates = {}
diffs = {}
for idx, name in enumerate(res.name):
    date = convertDate('_'.join(name.split('_')[3:5]))
    diff = (date - startDate).days
    if diff not in diffs.keys():
        diffs[diff] = []
    diffs[diff].append(idx)

# %%
for dayDiff, idx in diffs.items():
    labels = res.labels[idx]
    preds = res.preds[idx]

    acc = np.sum(labels == preds)/len(preds)
    print(f'{dayDiff}: {acc=}')
# %%
'classifySingleCellCrop-1715386736'