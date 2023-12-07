# %%
from src.models import trainBB
from src.data.fileManagement import getModelDetails
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

import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# %%
homePath = Path('../../')

datasetDictPathFull = homePath / 'data/TJ2201/split16/TJ2201DatasetDictNoBorderFull.npy'
datasetDictPathPartial = homePath / 'data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy'

datasetDicts = np.load(datasetDictPathFull, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]

# %%

def getDataloaders(modelName, homePath = homePath):
    modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
    outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
    if not outPath.exists():
        outPath = Path(str(outPath).replace('.out', '.txt'))
    assert outPath.exists(), outPath
    modelDetails = getModelDetails(outPath)
    print(modelDetails)
    model = trainBB.getTFModel(modelDetails['modelType'], modelPath)

    dataPath = Path.joinpath(homePath, 'data', modelDetails['experiment'], 'raw', 'phaseContrast')

    dataloaders, dataset_sizes = trainBB.makeImageDatasets(datasetDicts, 
                                                dataPath,
                                                modelDetails,
                                                isShuffle = False
                                                )
    return dataloaders, model


def gradCamModel(img, model, inputs, plotOn = True):
    img_numpy = img.numpy().transpose([1, 2, 0])
    img_numpy = (img_numpy - np.min(img_numpy))/np.ptp(img_numpy)
    # img_numpy = np.array([img_numpy, img_numpy, img_numpy])
    input_tensor = img.unsqueeze(0)

    target_layers = [model.layer4]
    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None
    
    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = GradCAMPlusPlus
    with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=True) as cam:


        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(img_numpy, grayscale_cam, use_rgb=True)
        cam_imageWrite = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    if plotOn:
        plt.figure()
        plt.subplot(121)
        plt.imshow(img_numpy)
        plt.subplot(122)
        plt.imshow(gb, cmap = 'jet')
        cv2.imwrite('./test.png', cam_imageWrite)

    return [img_numpy, cam_image, cam_gb, gb]
# %%
experiment  = 'TJ2201'
nIncrease   = 0
maxAmt      = 20000
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
optimizer = 'sgd'
notes = 'Run on coculture wells only'

modelID, idSource = 0, 0
modelSaveName = Path(f'../models/classification/classifySingleCellCrop-{modelID}.pth')
resultsSaveName = Path(f'../results/classificationTraining/classifySingleCellCrop-{modelID}.txt')


modelInputs = {

'experiment'    : experiment, 
'nIncrease'     : nIncrease,
'maxAmt'        : maxAmt,
'batch_size'    : batch_size,
'num_epochs'    : num_epochs,
'modelType'     : modelType,
'modelName'     : modelSaveName.parts[-1],
'modelIDSource' : 0,
'notes'         : notes,
'optimizer'     : optimizer, 
'augmentation'  : None
}

dataPath = Path.joinpath(homePath, 'data', modelInputs['experiment'], 'raw', 'phaseContrast')

imgs = {}
idx = 62
idx = 61
for nIncrease in [0, 25, 65]:
    modelInputs['nIncrease'] = nIncrease
    dataloaders, dataset_sizes = trainBB. makeImageDatasets(datasetDicts, 
                                                dataPath,
                                                modelInputs,
                                                data_transforms = None,
                                                isShuffle = False
                                                )
    np.unique(dataloaders['train'].dataset.phenotypes, return_counts=True)
    inputs, classes = next(iter(dataloaders['train']))
    img = inputs[idx]
    imgs[nIncrease] = img

    # if modelInputs['nIncrease'] == 25:
    #     imgInput = inputs[60].numpy().transpose((1,2,0))
plt.subplot(131)
plt.imshow(imgs[0].numpy().transpose((1,2,0)))
plt.subplot(132)
plt.imshow(imgs[25].numpy().transpose((1,2,0)))
plt.subplot(133)
plt.imshow(imgs[65].numpy().transpose((1,2,0)))
# %%
modelNames = { 0: 'classifySingleCellCrop-714689',   # 0 px increase
              25: 'classifySingleCellCrop-713279',   # 25 px increase
              65: 'classifySingleCellCrop-709125'}   # 65 px increase

pxExampleDict = {}
idx = 5
for pixelIncrease in modelNames.keys():
    img = imgs[pixelIncrease]
    modelName = modelNames[pixelIncrease]
    modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
    outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
    if not outPath.exists():
        outPath = Path(str(outPath).replace('.out', '.txt'))
    assert outPath.exists(), outPath
    modelDetails = getModelDetails(outPath)
    model = trainBB.getTFModel(modelDetails['modelType'], modelPath)

    gradImages = gradCamModel(img, model, inputs, plotOn = True)
    pxExampleDict[pixelIncrease] = gradImages
# %%
from src.data.fileManagement import collateModelParameters
dfExperiment = collateModelParameters(generate=True)


# %%
