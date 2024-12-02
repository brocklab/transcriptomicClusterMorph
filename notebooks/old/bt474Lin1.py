# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm
import pickle 

from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode
import detectron2

from src.models.trainBB import makeImageDatasets
from src.models import testBB
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset

from src.models import trainBB
from src.data.fileManagement import getModelDetails
from src.data.imageProcessing import imSplit
# %%
def getGreenRecord(datasetDicts, datasetDictsGreen = []):
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []

        for annotation in record['annotations']:
            annotation['bbox'] = detectron2.structures.BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            # if annotation['category_id'] == 1:
            #     newAnnotations.append(annotation)
        # if len(newAnnotations) > 0:
        #     record['annotations'] = newAnnotations
        #     datasetDictsGreen.append(record)
    return datasetDicts
# for record in tqdm(datasetDictsTreat):
#     for cell in record['annotations']:
#         cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
#         cell['bbox_mode'] = BoxMode.XYXY_ABS
# %%
datasetDictsGreen = {}

datasetDicts = load_coco_json('../../../data/TJ2342A/TJ2342ASegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2342A'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../../../data/TJ2442B/TJ2442BSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442B'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../../../data/TJ2442C/TJ2442CSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442C'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../../../data/TJ2442D/TJ2442DSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442D'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../../../data/TJ2442E/TJ2442ESegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442E'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../../../data/TJ2442F/TJ2442FSegmentationsGreenFiltered.json', '.')
datasetDictsGreen['TJ2442F'] = getGreenRecord(datasetDicts, [])
# %% Try to find dead cells
from skimage.measure import regionprops
from skimage.draw import polygon2mask

def getAreaEcc(polygon, imageShape):
    polyx = polygon[::2]
    polyy = polygon[1::2]
    polygonSki = list(zip(polyy, polyx))
    mask = polygon2mask(imageShape, polygonSki)
    reg = regionprops(mask.astype(np.uint8))

    if len(reg) > 0:
        area = reg[0].area
        eccentricity = reg[0].eccentricity
    
    else:
        area = 0
        eccentricity = 0

    return area, eccentricity
for experiment in datasetDictsGreen:
    datasetDicts = datasetDictsGreen[experiment]
    nAnnos = 0
    for record in datasetDicts:
        nAnnos += len(record['annotations'])
    print(f'{experiment} had {nAnnos} cells identified')

# %%
# allArea, allEcc, imgNames, allPoly = [], [], [], []

# # %%
# for experiment in datasetDictsGreen.keys():
#     datasetDicts = datasetDictsGreen[experiment]
#     print(experiment)
#     nAnnos = 0
#     for record in tqdm(datasetDicts):
#         image_shape = [record['height'], record['width']]
#         newAnnotations = []
#         for annotation in record['annotations']:
#             segmentation = annotation['segmentation'][0]

#             area, ecc = getAreaEcc(segmentation, image_shape)

#             if ecc > 0.8:
#                 newAnnotations.append(annotation)
#             allArea.append(area)
#             allEcc.append(ecc)
#             allPoly.append(segmentation)
#             imgNames.append(record['file_name'])
#         nAnnos += len(newAnnotations)
#         record['annotations'] = newAnnotations

#     print(f'{experiment} had {nAnnos} cells identified')
# # %%
# from detectron2.data import MetadataCatalog, DatasetCatalog
# import detectron2.data.datasets as datasets
    
# def getCells(datasetDict):
#     return datasetDict

# for experiment in datasetDictsGreen.keys():
#     fileName = f'../../../data/{experiment}/{experiment}SegmentationsGreenFiltered.json'
#     datasetDicts = datasetDictsGreen[experiment]
#     inputs = [datasetDicts]
#     if 'cellMorph' in DatasetCatalog:
#         DatasetCatalog.remove('cellMorph')
#         MetadataCatalog.remove('cellMorph')
#     DatasetCatalog.register("cellMorph", lambda x=inputs: getCells(inputs[0]))
#     MetadataCatalog.get("cellMorph").set(thing_classes=["cell"])
#     datasets.convert_to_coco_json('cellMorph', output_file=fileName, allow_cached=False)

# %%
def makeDatasets(modelInputs):

    experiments = datasetDictsGreen.keys()
    loaders = []
    for experiment in experiments:
        modelInputs['experiment'] = experiment
        dataPath = Path(f'../../../data/{experiment}/raw/phaseContrast')

        dataloader, dataset_sizes = makeImageDatasets(datasetDictsGreen[experiment], 
                                                    dataPath,
                                                    modelInputs,
                                                    phase = ['train']
                                                    )
        loaders.append(dataloader.dataset)

    sizeTrain = 0
    for cellDataset in loaders[0:-1]:
        sizeTrain += len(cellDataset)
    sizeTest = len(loaders[-1])

    loadersTrain = loaders[0:-1]
    loadersTest = [loaders[-1]]

    dataLoaderTrain = DataLoader(ConcatDataset(loadersTrain),
                                batch_size = modelInputs['batch_size'],
                                shuffle = True)


    dataLoaderTest = DataLoader(ConcatDataset(loadersTest),
                                batch_size = modelInputs['batch_size'],
                                shuffle = True)

    dataloaders = {'train': dataLoaderTrain, 'test': dataLoaderTest}
    dataset_sizes = {'train': len(dataLoaderTrain.dataset), 'test': len(dataLoaderTest.dataset)}

    return dataloaders


# %%


# %%
homePath = Path('../../../')
resultsFile = homePath / 'results' / 'classificationResults' / 'modelResultsCoCulture.pickle'
if resultsFile.exists():
    modelRes = pickle.load(open(resultsFile, "rb"))
else:
    modelRes = {}

modelNames = [
    'classifySingleCellCrop-1711380928',
    'classifySingleCellCrop-1711394245',
    'classifySingleCellCrop-1709844237',
    'classifySingleCellCrop-1711407593',

]

for modelName in modelNames:
    if modelName not in modelRes.keys():
        print(modelName)
        modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
        outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
        if not outPath.exists():
            outPath = Path(str(outPath).replace('.out', '.txt'))
        assert outPath.exists(), outPath
        modelInputs = getModelDetails(outPath)
        model = trainBB.getTFModel(modelInputs['modelType'], modelPath)

        print(modelName)
        dataloaders = makeDatasets(modelInputs)
        probs, allLabels, scores = testBB.testModel(model, dataloaders, mode = 'test')
        modelRes[modelName] = testBB.testResults(probs, allLabels, scores, modelName)

pickle.dump(modelRes, open(resultsFile, "wb"))
# %%
increases, aucs = [], []
for modelName in modelNames:
    outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
    if not outPath.exists():
        outPath = Path(str(outPath).replace('.out', '.txt'))
    assert outPath.exists(), outPath
    modelInputs = getModelDetails(outPath)
    res = modelRes[modelName]
    aucs.append(res.auc)
    increases.append(modelInputs['nIncrease'])
# %%
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.scatter(increases, aucs, s = 100)
plt.plot(increases, aucs)
plt.xticks(increases)
plt.xlabel('Pixel Increase')
plt.ylabel('AUC')
plt.savefig('../../../figures/publication/results/increasingBBLineage.png', dpi = 500, bbox_inches = 'tight')
# %%
# Find green cell
datasetDicts = datasetDictsGreen['TJ2442F']
newDatasetDicts = []
for record in tqdm(datasetDicts):
    record = record.copy()
    newAnnotations = []

    for annotation in record['annotations']:
        annotation['bbox'] = detectron2.structures.BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
        annotation['bbox_mode'] = BoxMode.XYXY_ABS
        if annotation['category_id'] == 1:
            newAnnotations.append(annotation)
    if len(newAnnotations) > 0:
        record['annotations'] = newAnnotations
        newDatasetDicts.append(record)
# %%
imgDir = homePath / 'data/TJ2442F/raw/phaseContrast'
# imgIdx = 13
# imgIdx = 1
imgIdx = 300
cellIdx = 0
record = newDatasetDicts[imgIdx]
annotation = record['annotations'][cellIdx]
seg = annotation['segmentation'][0]
polyx = seg[::2]
polyy = seg[1::2]

img = homePath / record['file_name']
imgName = str(img.name)
imgFull = imgName.split('_')
imgNum = int(imgFull[-1].split('.png')[0])
imgFull = '_'.join(imgFull[0:-1]) + '.png'
phaseContrastFull = imread(imgDir / imgFull)

compositeDir = Path(str(imgDir).replace('phaseContrast', 'composite'))
imgFullComposite = str(imgFull).replace('phaseContrast', 'composite')
compositeFull = imread(compositeDir / imgFullComposite)

compositeSplit = imSplit(compositeFull, nIms = 4)

compositeIm = compositeSplit[imgNum - 1][:,:,0:3]
plt.imshow(compositeIm)
plt.plot(polyx, polyy, c = 'red')

plt.figure()
from src.data.imageProcessing import segmentGreenHigh
nGreen, BW = segmentGreenHigh(compositeIm)
plt.imshow(BW)
# imsave('../../../figures/tempPres/compositeGreen.png', compositeIm)
# %%
# from skimage import color, morphology
# se = morphology.disk(2)
# res = morphology.white_tophat(compositeIm[:,:,1], se)

# compositeImTh = compositeIm.copy()
# compositeImTh[:,:,1] = res
# plt.imshow(res)
# %%
