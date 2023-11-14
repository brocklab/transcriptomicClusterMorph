# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.data.fileManagement import convertDate
from src.models import modelTools
from src.data.imageProcessing import bbIncrease
from src.data.fileManagement import splitName2Whole

import os
from pathlib import Path
import numpy as np
import sys
import argparse 
import matplotlib.pyplot as plt
from skimage.io import imread
# %% Add argparse
parser = argparse.ArgumentParser(description='Network prediction parameters')
parser.add_argument('--experiment', type = str, metavar='experiment',  help = 'Experiment to run')
parser.add_argument('--nIncrease',  type = int, metavar='nIncrease',   help = 'Increase of bounding box around cell')
parser.add_argument('--maxAmt',     type = int, metavar='maxAmt',      help = 'Max amount of cells')
parser.add_argument('--batch_size', type = int, metavar='batch_size',  help = 'Batch size')
parser.add_argument('--num_epochs', type = int, metavar='num_epochs',  help = 'Number of epochs')
parser.add_argument('--modelType',  type = str, metavar='modelType',   help = 'Type of model (resnet, vgg, etc.)')
parser.add_argument('--notes',      type = str, metavar='notes',       help = 'Notes on why experiment is being run')
parser.add_argument('--optimizer',  type = str, metavar='optimizer',   help = 'Optimizer type')
parser.add_argument('--augmentation',  type = str, metavar='augmentation',   help = 'Image adjustment (None, blackoutCell, stamp)')

# This is for running the notebook directly
args, unknown = parser.parse_known_args()

# %%
experiment  = 'TJ2201'
nIncrease   = 0
maxAmt      = 20000
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
optimizer = 'sgd'
notes = 'Run on coculture wells only'

modelID, idSource = modelTools.getModelID(sys.argv)
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
'modelIDSource' : idSource,
'notes'         : notes,
'optimizer'     : optimizer, 
'augmentation'  : None
}

argItems = vars(args)

for item, value in argItems.items():
    if value is not None:
        print(f'Replacing {item} value with {value}')
        modelInputs[item] = value
modelDetailsPrint = modelTools.printModelVariables(modelInputs)


# %%
dataPath = Path(f'../../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
datasetDicts = [record for record in datasetDicts if len(record['annotations']) > 0]
# %%
# Check size
wellSize = {}
for seg in datasetDicts:
    well = seg['file_name'].split('_')[1]
    if well not in wellSize.keys():
        wellSize[well] = 0
    wellSize[well] += len(seg['annotations'])
print(np.array(list(wellSize.values())).sum(axis=0))
# %%
imgs = []
for nIncrease in [0, 25, 65]:
    modelInputs['nIncrease'] = nIncrease
    dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                                dataPath,
                                                modelInputs,
                                                data_transforms = None,
                                                isShuffle = False
                                                )
    np.unique(dataloaders['train'].dataset.phenotypes, return_counts=True)
    inputs, classes = next(iter(dataloaders['train']))
    img = inputs[61].numpy().transpose((1,2,0))
    imgs.append(img)

    if modelInputs['nIncrease'] == 25:
        imgInput = inputs[60].numpy().transpose((1,2,0))

# %%
plt.imshow(imgInput)
plt.axis('off')
plt.savefig('../../figures/publication/exemplar/inputImage_dataloder.png', bbox_inches='tight', pad_inches=0, transparent = True)
# %%
plt.figure(figsize = (11, 10))
plt.subplot(131)
plt.imshow(imgs[0])
plt.axis('off')
plt.title('0 px\nIncrease')
plt.subplot(132)
plt.imshow(imgs[1])
plt.axis('off')
plt.title('25 px\nIncrease')
plt.subplot(133)
plt.imshow(imgs[2])
plt.axis('off')
plt.title('65 px\nIncrease')
plt.savefig('../../figures/publication/exemplar/increasingBB_dataloader.png', bbox_inches='tight', pad_inches=0.1, transparent = True)
# %%
imgs = []
modelInputs['nIncrease'] = 25
for augmentation in ['None', 'blackoutCell', 'stamp']:
    modelInputs['augmentation'] = augmentation
    dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                                dataPath,
                                                modelInputs,
                                                data_transforms = None,
                                                isShuffle = False
                                                )
    np.unique(dataloaders['train'].dataset.phenotypes, return_counts=True)
    inputs, classes = next(iter(dataloaders['train']))
    img = inputs[61].numpy().transpose((1,2,0))
    imgs.append(img)
# %%
plt.figure(figsize = (11, 10))
plt.subplot(131)
plt.imshow(imgs[0])
plt.axis('off')
plt.title('No Augmentation')
plt.subplot(132)
plt.imshow(imgs[1])
plt.axis('off')
plt.title('No Texture')
plt.subplot(133)
plt.imshow(imgs[2])
plt.axis('off')
plt.title('No Surrounding')
plt.savefig('../../figures/publication/exemplar/augmentations_dataloader.png', bbox_inches='tight', pad_inches=0.1, transparent = True)
# %%
imgIdx = 20
cellIdx = 2

annotations = datasetDicts[imgIdx]['annotations']
imgName = datasetDicts[imgIdx]['file_name'].split('/')[-1]
imgNameWhole = splitName2Whole(imgName)
fullPath = os.path.join('../../data/TJ2201/raw/phaseContrast', imgNameWhole)
img = imread(fullPath)

bb = annotations[cellIdx]['bbox']
poly = annotations[cellIdx]['segmentation'][0]
poly = np.array(np.reshape(poly, (int(len(poly)/2), 2)))
imgCrop = bbIncrease(poly, bb, imgName, img, 16, nIncrease = 25)
plt.imshow(imgCrop, cmap = 'gray')
plt.axis('off')
plt.savefig('../../figures/publication/exemplar/inputImage_image.png', bbox_inches='tight', pad_inches=0, transparent = True)
# %%
imgIdx = 20
cellIdx = 11

annotations = datasetDicts[imgIdx]['annotations']
imgName = datasetDicts[imgIdx]['file_name'].split('/')[-1]
imgNameWhole = splitName2Whole(imgName)
fullPath = os.path.join('../../data/TJ2201/raw/phaseContrast', imgNameWhole)
img = imread(fullPath)

bb = annotations[cellIdx]['bbox']
poly = annotations[cellIdx]['segmentation'][0]
poly = np.array(np.reshape(poly, (int(len(poly)/2), 2)))
imgs = []
for nIncrease in [0, 25, 65]:
    imgCrop = bbIncrease(poly, bb, imgName, img, 16, nIncrease)
    imgs.append(imgCrop)
# %%

fig = plt.figure(constrained_layout=True, figsize=(10, 4))

# create 3 subfigs (width padding=30%)
sf1, sf2, sf3 = fig.subfigures(1, 3, wspace=0.1)

# add an axes to each subfig (left=0%, bottom=0%, width=100%, height=90%)
ax1 = sf1.add_axes([0, 0, 1, 0.9])
ax2 = sf2.add_axes([0, 0, 1, 0.9])
ax3 = sf3.add_axes([0, 0, 1, 0.9])

ax1.imshow(imgs[0], cmap = 'gray')
ax1.axis('off')

ax2.imshow(imgs[1], cmap = 'gray')
ax2.axis('off')

ax3.imshow(imgs[2], cmap = 'gray')
ax3.axis('off')

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')

sf1.suptitle('0 px\nIncrease',  y = .9)
sf2.suptitle('25 px\nIncrease', y = .9)
sf3.suptitle('65 px\nIncrease', y = .9)
# plt.subplots_adjust(top = 0.95)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('../../figures/publication/exemplar/increasingBB_image.png', bbox_inches='tight', pad_inches=0.1, transparent = True)

# %%
