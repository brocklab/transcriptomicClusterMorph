# %%
import pickle
import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
resNameTreat = 'classifySingleCellCrop-1715810868'

modelNames = [resName231, resName436, resNameTreat]
modelNames = {'MDA-MB-231 Subpopulations': resName231,
              'MDA-MB-436 Subpopulations': resName436,
              'Treated Populations': resNameTreat}
# %%
homePath = Path('../../../')
resultsFile = homePath / 'results' / 'classificationResults' / 'modelResultsCoCulture.pickle'
if resultsFile.exists():
    modelRes = pickle.load(open(resultsFile, "rb"))
# %%
def getDayAcc(modelName):
    res = modelRes[modelName]
    # Get image names dependent on experiment
    if modelName.endswith('868'):
        imgNames = res.imgNames
    else:
        imgNames = res.name
    allDates = [convertDate('_'.join(name.split('_')[3:5])) for name in imgNames]
    startDate, endDate = min(allDates), max(allDates)
    idxDates = {}
    diffs = {}
    for idx, name in enumerate(imgNames):
        date = convertDate('_'.join(name.split('_')[3:5]))
        diff = (date - startDate).days
        if diff not in diffs.keys():
            diffs[diff] = []
        diffs[diff].append(idx)

    dayAcc = {}
    for dayDiff, idx in diffs.items():
        labels = res.labels[idx]
        preds = res.preds[idx]

        acc = np.sum(labels == preds)/len(preds)
        print(f'{dayDiff}: {acc=}')
        dayAcc[dayDiff] = acc
    return dayAcc
# %%
experimentDayAcc = {}
c = 0
colors = ['red', 'blue', 'green']
plt.figure()
for experimentName, modelName in modelNames.items():
    print(experimentName)
    dayAcc = getDayAcc(modelName)

    if 'Treat' in experimentName:
        dayAcc[0] = (dayAcc[0] + dayAcc[257])/2
        dayAcc[1] = (dayAcc[1] + dayAcc[258])/2
        dayAcc[2] = (dayAcc[2] + dayAcc[259])/2
        dayAcc[3] = (dayAcc[3] + dayAcc[260])/2

        dayAcc.pop(257)
        dayAcc.pop(258)
        dayAcc.pop(259)
        dayAcc.pop(260)

    days = list(dayAcc.keys())
    accs = list(dayAcc.values())
    da = list(zip(days, accs))
    da.sort()
    days = [d for d, a in da]
    accs = [a for i, a in da]
    plt.scatter(days, accs, c=colors[c], label=experimentName)
    plt.plot(days, accs, c=colors[c])
    c += 1    

plt.xticks([0, 1, 2, 3])
plt.xlabel('Day of Observation')
plt.ylabel('Accuracy')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25))
# %%
