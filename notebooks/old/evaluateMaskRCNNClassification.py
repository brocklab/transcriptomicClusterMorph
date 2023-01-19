# %%
import cellMorphHelper

import torch
import detectron2

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random, pickle
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

# Import image processing
from skimage import measure
from skimage import img_as_float
from skimage.io import imread
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border
from scipy.spatial import ConvexHull

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
# %%
def findFluorescenceColor(RGB, mask):
    """
    Finds the fluorescence of a cell
    Input: RGB image location
    Output: Color
    """
    # RGB = imread(RGBLocation)
    mask = mask.astype('bool')
    RGB[~np.dstack((mask,mask,mask))] = 0
    nGreen, BW = cellMorphHelper.segmentGreen(RGB)
    nRed, BW = cellMorphHelper.segmentRed(RGB)
    if nGreen>=(nRed+100):
        return "green"
    elif nRed>=(nGreen+100):
        return "red"
    else:
        return "NaN"
# %% Getting/reading *.npy segmentations
def getCells(experiment, imgType, stage=None):
    segDir = os.path.join('../data',experiment,'segmentedIms')
    segFiles = os.listdir(segDir)
    segFiles = [segFile for segFile in segFiles if segFile.endswith('.npy')]
    idx = 0

    if stage in ['train', 'test']:
        print(f'In {stage} stage')
        random.seed(1234)
        random.shuffle(segFiles)
        trainPercent = 0.75
        trainNum = int(trainPercent*len(segFiles))
        if stage == 'train':
            segFiles = segFiles[0:trainNum]
        elif stage == 'test':
            segFiles = segFiles[trainNum:]

    datasetDicts = []
    for segFile in tqdm(segFiles):

        # Load in cellpose output
        segFull = os.path.join(segDir, segFile)
        seg = np.load(segFull, allow_pickle=True)
        seg = seg.item()

        splitMasks = cellMorphHelper.imSplit(seg['masks'])
        nSplits = len(splitMasks)

        splitDir = f'{experiment}Split{nSplits}'
        imgBase = cellMorphHelper.getImageBase(seg['filename'].split('/')[-1])    
        for splitNum in range(1, len(splitMasks)+1):
            imgFile = f'{imgType}_{imgBase}_{splitNum}.png'
            imgFileComposite = f'composite_{imgBase}_{splitNum}.png'

            imgPath = os.path.join('../data', splitDir, imgType, imgFile)
            compositePath = os.path.join('../data', splitDir, 'composite', imgFileComposite)
            assert os.path.isfile(imgPath)
            composite = imread(compositePath)
            record = {}
            record['file_name'] = imgPath
            record['image_id'] = idx
            record['height'] = splitMasks[splitNum-1].shape[0]
            record['width'] = splitMasks[splitNum-1].shape[1]

            mask = splitMasks[splitNum-1]
            cellNums = np.unique(mask)
            cellNums = cellNums[cellNums != 0]

            cells = []
            for cellNum in cellNums:
                cellMask = mask == cellNum
                color = findFluorescenceColor(composite.copy(), cellMask.copy())

                if color == 'red': # ESAM Negative
                    catID = 0
                elif color == 'green': # ESAM Positive
                    catID = 1
                else:
                    continue
                contours = measure.find_contours(img_as_float(mask==cellNum), .5)
                fullContour = np.vstack(contours)

                px = fullContour[:,1]
                py = fullContour[:,0]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                if len(poly) < 4:
                    return
                cell = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": catID,
                }
                cells.append(cell)
            record["annotations"] = cells  
            datasetDicts.append(record)
            idx+=1
    return datasetDicts
# %% Build model
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
if not torch.cuda.is_available():
    print('CUDA not available, resorting to CPU')
    cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cellMorph_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (cell). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = '../models/TJ2201Split16Classify'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
predictor = DefaultPredictor(cfg)
# %% Load all images
experiment = 'TJ2201'
imgType = 'phaseContrast'
# %%
saveData = 0
if saveData:
    inputs = [experiment, imgType]
    datasetDicts = getCells(inputs[0], inputs[1], stage='test')
    allOutputs, images = [], []
    for i,d in tqdm(enumerate(datasetDicts)):
        im = cv2.imread(d["file_name"])
        allOutputs.append(predictor(im))
        images.append(d['file_name'])
        if i%10 == 0:
            np.save('../data/esamMonoSegmented/maskRCNNResults.npy', [allOutputs, images])
    np.save('../data/esamMonoSegmented/maskRCNNResults.npy', [allOutputs, images])
else:
    predictorOutput = np.load('../data/esamMonoSegmented/maskRCNNResults.npy', allow_pickle=True)
# %%
# D2 - ESAM + Monoculture
# E2 - ESAM - Monoculture
# E7 - Coculture
ESAMPosActual, ESAMPosLabel = 0, 0
ESAMNegActual, ESAMNegLabel = 0, 0
ESAMPosScore, ESAMNegScore = [], []
for output, imgPhase in zip(predictorOutput[0], predictorOutput[1]):
    imgComposite = imgPhase.replace('phaseContrast', 'composite')
    well = imgPhase.split('_')[1]
    

    img = imread(imgComposite)

    # Threshold out low quality cells
    minThresh = 0.9
    instances = output['instances']
    idxDelete = [idx for idx, score in enumerate(instances.scores) if score<minThresh]

    scores = np.delete(instances.scores, idxDelete)
    pred_classes = np.delete(instances.pred_classes, idxDelete)
    pred_masks = np.delete(instances.pred_masks.numpy(), idxDelete, axis=0)

    nPreds = len(scores)
    if well == 'E2': # ESAM -, class 0
        ESAMNegActual += nPreds
        ESAMNegLabel += np.sum(pred_classes.numpy() == 0)
    elif well == 'D2': # ESAM +, class 1
        ESAMPosActual += nPreds
        ESAMPosLabel += np.sum(pred_classes.numpy() == 1)

    # Find accuracy 
# # %%
# nIms = 1
# imSpace = 1
# lines = []
# random.seed(1234)
# for d in random.sample(datasetDicts, 3): 
#     im = cv2.imread(d["file_name"])
#     pcName = d["file_name"]
#     compositeName = pcName.replace('phaseContrast', 'composite')

#     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=MetadataCatalog.get('allCells_test'),
#                    scale=1,
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     plt.figure(figsize=(20,10))
#     plt.subplot(121)
#     plt.imshow(out.get_image()[:, :, ::-1])
#     # plt.title(d['file_name'])
#     plt.subplot(122)
#     imComposite = imread(compositeName)
#     plt.imshow(imComposite)

# # %%

# # %%
# wells = []
# for i, d in enumerate(datasetDicts):
#     fname = d['file_name']
#     fname = fname.split('_')[1]
#     wells.append(fname)
#     if fname == 'D2':
#         print(i)
# print(set(wells))

# # %%
# d = datasetDicts[88]
# im = cv2.imread(d["file_name"])
# pcName = d["file_name"]
# compositeName = pcName.replace('phaseContrast', 'composite')

# outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
# v = Visualizer(im[:, :, ::-1],
#                 metadata=MetadataCatalog.get('allCells_test'),
#                 scale=1,
#                 instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
# )
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# plt.figure(figsize=(20,10))
# plt.subplot(121)
# plt.imshow(out.get_image()[:, :, ::-1])
# # plt.title(d['file_name'])
# plt.subplot(122)
# imComposite = imread(compositeName)
# plt.imshow(imComposite)
# # %%
# masks = outputs['instances'].pred_masks
# plt.imshow(masks[-2])
# # %%
# from skimage.measure import label
# from skimage.color import label2rgb
# minThresh = 0.9
# instances = outputs['instances']
# idxDelete = [idx for idx, score in enumerate(instances.scores) if score<minThresh]

# scores = np.delete(instances.scores, idxDelete)
# pred_classes = np.delete(instances.pred_classes, idxDelete)
# pred_masks = np.delete(instances.pred_masks.numpy(), idxDelete, axis=0)

# # Plotting outputs
# fullMask = np.zeros(np.shape(pred_masks[0]))
# cellNum = 1
# for mask in pred_masks:
#     isCell = np.where(mask == True)
#     fullMask[isCell[0], isCell[1]] = cellNum
#     cellNum += 1
# labelImg = label(fullMask)
# label_overlay = label2rgb(labelImg, im)
# plt.imshow(label_overlay)
