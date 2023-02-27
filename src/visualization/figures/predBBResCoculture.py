# %% [markdown[]
"""

"""
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


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

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
datasetDictPath = homePath / 'data/TJ2201/split16/TJ2201DatasetDictNoBorderOrig.npy'
# %%
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]

# %%
resultsFile = homePath / 'results' / 'classificationResults' / 'modelResultsCoCulture.pickle'
if resultsFile.exists():
    modelRes = pickle.load(open(resultsFile, "rb"))
else:
    modelRes = {}
for modelName in modelNames:
    if modelName not in modelRes.keys():
        print(modelName)
        probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts)
        modelRes[modelName] = testBB.testResults(probs, allLabels, scores, imgNames, modelName)
        pickle.dump(modelRes, open(resultsFile, "wb"))
# %%
modelsPlot = ['classifySingleCellCrop-714689',
              'classifySingleCellCrop-713279', 
              'classifySingleCellCrop-709125', 
              'classifySingleCellCrop-720396',
              'classifySingleCellCrop-723948']
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for model in modelsPlot:
    modelDetails = testBB.getModelDetails(homePath / 'results' / 'classificationTraining' / f'{model}.out')
    res = modelRes[model]
    auc = res.auc
    plotLabel = f'BB increase {modelDetails["nIncrease"]} px, AUC = {auc:0.2f}'
    plt.plot(res.fpr, res.tpr, label=plotLabel, linewidth=3)
plt.legend(fontsize=12)
plt.title('Phenotype Prediction\nIncreasing Bounding Box')
# plt.savefig(homePath / 'figures' / 'bbIncreaseCocultureROC.png', dpi=600)
# %%
x = plotTrainingRes(homePath / 'results' / 'classificationTraining' / f'{modelNames[-1]}.out')
# # %% Identify images that were not in the training set
# modelName = 'classifySingleCellCrop-713279'
# modelDetails = testBB.getModelDetails(homePath / 'results' / 'classificationTraining' / f'{modelName}.out')
# data_transforms = {
#     'train': transforms.Compose([
#         # transforms.RandomResizedCrop(224),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean, std)
#     ])
# }
# dataPath = Path(homePath / f'data/{modelDetails["experiment"]}/raw/phaseContrast')
# image_datasetTrain = singleCellLoader(datasetDicts, 
#                                 data_transforms['train'], 
#                                 dataPath, 
#                                 nIncrease = modelDetails['nIncrease'], 
#                                 phase='train', 
#                                 maxAmt = modelDetails['maxAmt'])
# trainImgs = list(set(image_datasetTrain.imgNames))
# # wellsSave = ['E2', 'D2', 'E7']
# datasetDictsSub = []
# for seg in tqdm(datasetDicts):
#     imgName = Path(seg['file_name']).parts[-1]
#     well = seg['file_name'].split('_')[1]
#     if imgName not in trainImgs:
#         datasetDictsSub.append(seg)
# # %%
# modelPath = homePath / 'models' / 'classification' / modelDetails['modelName']
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# model = getTFModel(modelDetails['modelType'])
# model.load_state_dict(torch.load(modelPath, map_location=device))
# # %%

# image_dataset = singleCellLoader(datasetDictsSub, 
#                                 data_transforms['train'], 
#                                 dataPath, 
#                                 nIncrease = modelDetails['nIncrease'], 
#                                 phase='none', 
#                                 maxAmt = 0)
# dataset_sizes = len(image_dataset)
# dataloader = DataLoader(image_dataset, batch_size=modelDetails['batch_size'], shuffle=False)

# device_str = "cuda"
# device = torch.device(device_str if torch.cuda.is_available() else "cpu")
# probs = []
# allLabels = []
# scores = []
# running_corrects = 0
# for inputs, labels in tqdm(dataloader, position=0, leave=True):
#     # I have no idea why you have to do this but...
#     # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-float-for-argument-2-weight/38961/9
#     inputs = inputs.float()
#     inputs = inputs.to(device)
#     labels = labels.to(device)


#     outputs = model(inputs)
#     _, preds = torch.max(outputs, 1)
#     probs.append(outputs.cpu().data.numpy())
#     allLabels.append(labels.cpu().data.numpy())
#     scores.append(F.softmax(outputs, dim=1).cpu().data.numpy())
#     running_corrects += torch.sum(preds == labels.data)
    

# probs = np.concatenate(probs)
# allLabels = np.concatenate(allLabels)
# scores = np.concatenate(scores)
# allPreds = np.argmax(scores, axis= 1)
# modelTestResults = {'probs': probs, 'allPreds': allPreds, 'allLabels': allLabels, 'scores': scores, 'images': dataloader.dataset.imgNames}
# fpr, tpr, _ = roc_curve(modelTestResults['allLabels'], modelTestResults['scores'][:,0])
# roc_auc = roc_auc_score(modelTestResults['allLabels'], modelTestResults['scores'][:,0])
# print(f'AUC: {roc_auc:0.3}')
# # %%
# allDates = []
# for image in modelTestResults['images']:
#     date = '_'.join(image.split('_')[3:5])
#     allDates.append(convertDate(date))
# allDates = np.array(allDates)
# uniqueDates = list(set(allDates))
# uniqueDates.sort()
# for date in uniqueDates:
#     dateIdx = np.where(allDates == date)[0]
#     currentLabels = allLabels[dateIdx]
#     currentScores = scores[dateIdx]
#     currentPreds = np.argmax(scores[dateIdx], axis=1)
#     acc = np.sum(currentPreds == currentLabels)/len(currentLabels)
#     roc_auc = roc_auc_score(currentLabels, currentScores[:,1])
#     print(f'AUC: {roc_auc:0.2f} \t Accuracy: {acc:0.2f}')
# # %%
# groundTruth = {}
# for seg in datasetDictsSub:
#     fileName = splitName2Whole(Path(seg['file_name']).parts[-1])
#     if fileName not in groundTruth.keys():
#         groundTruth[fileName] = [0, 0]
#     for annotation in seg['annotations']:
#         category_id = annotation['category_id']
#         groundTruth[fileName][category_id] += 1

# predictedLabels = np.argmax(modelTestResults['scores'], axis=1)
# predicted, groundTruth = {}, {}
# for predLabel, imgName, trueLabel in zip(predictedLabels, modelTestResults['images'], modelTestResults['allLabels']):
#     imgName = splitName2Whole(imgName)
#     if imgName not in predicted.keys():
#         predicted[imgName] = [0, 0]
#         groundTruth[imgName] = [0, 0]
#     predicted[imgName][predLabel] += 1
#     groundTruth[imgName][trueLabel] += 1

# # %%
# dfPred = pd.DataFrame(predicted).transpose()
# dfTrue = pd.DataFrame(groundTruth).transpose()

# dfFull = dfPred.join(dfTrue, lsuffix='_pred', rsuffix='_true')

# wells = [img.split('_')[1] for img in dfFull.index]
# dates = [convertDate('_'.join(img.split('_')[3:5])) for img in dfFull.index]
# dfFull['well'] = wells
# dfFull['dates'] = dates
# dfFull.head()

# # %%
# def getWell(dfFull, well):
#     """Extracts data for only specified well"""
#     if well != 'all':
#         dfWell = dfFull[dfFull['well'] == well]
#     else:
#         dfWell = dfFull.copy()
#     dfWell = dfWell.sort_values(by=['dates'])
#     dfWell = pd.DataFrame(dfWell.groupby('dates').sum()).reset_index()
#     return dfWell

# def ccc(x,y):
#     vx, cov_xy, cov_xy, vy = np.cov(x,y, bias=True).flat
#     mx, my = x.mean(), y.mean()
#     return 2*cov_xy / (vx + vy + (mx-my)**2)

# # %%
# dfWell = getWell(dfFull, 'all')   
# plt.scatter(dfWell['0_pred'], dfWell['0_true'],  color='green')
# plt.scatter(dfWell['1_pred'], dfWell['1_true'],  color='red')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')

# slope, intercept, r_value0, p_value, std_err = stats.linregress(dfWell['0_pred'], dfWell['0_true'])
# slope, intercept, r_value1, p_value, std_err = stats.linregress(dfWell['1_pred'], dfWell['1_true'])

# plt.title(f'R^2 = {r_value0**2:.3}')
# # %% 
# vals = np.random.randint(2, size=30000)
# predVals = []
# opposite = {0: 1, 1:0}
# for val in vals:
#     roll = np.random.random()
#     if roll > .9:
#         predVals.append(opposite[val])
#     else:
#         predVals.append(val)

# df = pd.DataFrame([vals, predVals]).transpose()
# N = 60
# df = df.groupby(df.index // N).sum()

# rvalue = ccc(df[0], df[1])
# plt.scatter(df[0], df[1])
# plt.title(rvalue)
# # %%
# roc_auc = roc_auc_score(modelTestResults['allLabels'], modelTestResults['scores'][:,0])