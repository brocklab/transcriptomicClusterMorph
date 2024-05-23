# %%
from src.data.fileManagement import collateModelParameters
from src.visualization.trainTestRes import plotTrainingRes
import pandas as pd
from pathlib import Path
# %%
dfModels = collateModelParameters(generate=True)
# %%
cond = (dfModels['experiment'] == 'TJ2453-436Co') & (dfModels['maxAmt'] == 200000000000) & (dfModels['batch_size'] == 32)
dfModels = dfModels.loc[cond]
dfModels.tail()
# %%
resPath = Path('../results/classificationTraining')
modelIncreaseEpochs = {}
finalModelDict = {}
for modelName, nIncrease in zip(dfModels['modelName'], dfModels['nIncrease']):
    modelName = modelName.replace('.pth', '.txt')
    modelPath = resPath / modelName
    trainRes = plotTrainingRes(modelPath, plot=False, title='', homePath = '')
    nEpochs = len(trainRes[0])

    if nIncrease not in modelIncreaseEpochs.keys():
        modelIncreaseEpochs[nIncrease] = nEpochs
        finalModelDict[nIncrease] = modelName
    elif nEpochs > modelIncreaseEpochs[nIncrease]:

        modelIncreaseEpochs[nIncrease] = nEpochs
        finalModelDict[nIncrease] = modelName
# %%
dfModels2 = dfModels[dfModels['nIncrease'] == 5]
for modelName in dfModels2['modelName']:
    modelName = modelName.replace('.pth', '.txt')
    modelPath = resPath / modelName
    trainRes = plotTrainingRes(modelPath, plot=False, title='', homePath = '')
    trainLoss, trainAcc, testLoss, testAcc = trainRes

    if len(testAcc) == 0:
        continue
    print(f'{modelName}: {max(testAcc):0.2f}\t{len(testLoss)}')