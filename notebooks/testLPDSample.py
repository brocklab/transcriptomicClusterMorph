# %%
from src.models.trainBB import makeImageDatasets, getTFModel
from src.models import testBB, trainBB
from src.data.fileManagement import getModelDetails
from src.models import modelTools
from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import torch.nn as nn
import torch
import torch.optim as optim


# %%
experiment  = 'TJ2302'
nIncrease   = 10
maxAmt      = 9e9
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
notes = 'Full test with Adam'

modelID, idSource = modelTools.getModelID(sys.argv)
modelSaveName = Path(f'../models/classification/classifySingleCellCrop-{modelID}.pth')
resultsSaveName = Path(f'../results/classificationTraining/classifySingleCellCrop-{modelID}.txt')

experiment = 'TJ2322-LPD7'

modelInputs = {

'experiment'    : experiment, 
'nIncrease'     : nIncrease,
'maxAmt'        : maxAmt,
'batch_size'    : batch_size,
'num_epochs'    : num_epochs,
'modelType'     : modelType,
'modelName'     : modelSaveName.parts[-1],
'modelIDSource' : idSource,
'notes'         : notes,
'augmentation'  : None

}
# %%
import json
experimentParamsLoc = '/home/user/work/cellMorph/data/experimentParams.json'
with open(experimentParamsLoc, 'r') as json_file:
    experimentParams = json.load(json_file)
experimentParams['TJ2322-LPD7'] = experimentParams['TJ2303-LPD4'].copy()
experimentJson = json.dumps(experimentParams)
with open(experimentParamsLoc, 'w') as json_file:
    json_file.write(experimentJson)
# %%
datasetDictPath = Path(f'../data/{experiment}/{experiment}DatasetDicts-1.npy')
datasetDicts = list(np.load(datasetDictPath, allow_pickle=True))
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
# %%
wellSize = {}
for seg in datasetDicts:
    well = seg['file_name'].split('_')[1]
    if well not in wellSize.keys():
        wellSize[well] = 0
    wellSize[well] += len(seg['annotations'])
# %%
dataloader, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath,
                                               modelInputs,
                                               phase=['none']
                                            )
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not modelSaveName.parent.exists():
    raise NotADirectoryError('Model directory not found')

model = getTFModel(modelInputs['modelType'])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
modelName = 'classifySingleCellCrop-1700026902'
modelName = 'classifySingleCellCrop-1700187095'
homePath = Path('../')
modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
if not outPath.exists():
    outPath = Path(str(outPath).replace('.out', '.txt'))
assert outPath.exists(), outPath
modelInputs = getModelDetails(outPath)
modelInputs['augmentation'] = None
print(modelInputs)
model = trainBB.getTFModel(modelInputs['modelType'], modelPath)

dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

modelInputs['experiment'] = 'TJ2322-LPD7'

allWells, allProps = [], []
for testWell in wellSize.keys():
    modelInputs['testWell'] = testWell
    dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                                dataPath,
                                                modelInputs,
                                                isShuffle = False
                                                )

    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model.to(device)
    allPreds = []
    for inputs, labels in tqdm(dataloaders['test']):
        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        allPreds.append(preds.cpu().numpy())
        print(sum(preds)/len(preds))
    preds = np.concatenate(allPreds)
    prop = sum(preds)/len(preds)*100

    allWells.append(testWell)
    allProps.append(prop)
    pd.DataFrame([allWells, allProps]).to_csv('./lpdSample.csv')

    print(f'Test well: {testWell} = {prop:0.2f}%')
# %% Test on other dataset
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns

lpdSample = pd.read_csv('../data/misc/lpdSample-new.csv', header = None, index_col=0).T
lpdSample = pd.read_csv('./lpdSample.csv', header = None, index_col=0).T
lpdSample.columns = [0, 'well', 'proportion']
lpdSample['proportion'] = lpdSample['proportion'].astype('float')
lpdSample['model'] = 'new'
# %%
plt.hist(lpdSample['proportion'], bins='doane')
plt.xlabel('Predicted Proportion')
plt.ylabel('Amount')
plt.axvline(28.55, color = 'r', label = 'Abundance Seq')
plt.axvline(np.mean(lpdSample['proportion']), color = 'magenta', label = 'Mean Imaging Proportion')
plt.legend()
# %%

# %%
