# %%
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
import os, cv2, random
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

# Import image processing
from skimage import measure
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border
from scipy.spatial import ConvexHull

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from src.data import imageProcessing, fileManagement
# %% Working with split images
experiment = 'TJ2406-MDAMB436-20X'
imgType = 'phaseContrast'
stage = 'train'

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
    homePath = Path('../data')
    for segFile in tqdm(segFiles):

        # Load in cellpose output
        segFull = os.path.join(segDir, segFile)
        seg = np.load(segFull, allow_pickle=True)
        seg = seg.item()

        # Load image
        imgBase = fileManagement.getImageBase(str(seg['filename']).split('/')[-1])    
        imgFile = homePath / experiment / 'raw' / 'phaseContrast' / f'phaseContrast_{imgBase}.png' 

        img = imread(imgFile)

        splitIms = imageProcessing.imSplit(img)
        splitMasks = imageProcessing.imSplit(seg['masks'])
        nSplits = len(splitMasks)

        splitDir = f'{experiment}/split{nSplits}'
        splitNum = 1
        for splitMask, splitImg in zip(splitMasks, splitIms):
            splitImgPath = f'{str(imgFile)[0:-4]}_{splitNum}.png'.replace('raw', 'split16')
            imgFile = f'{imgType}_{imgBase}_{splitNum}.png'
            imgPath = os.path.join(Path('../data'), splitDir, imgType, imgFile)
            imsave(imgPath, splitImg)
            assert os.path.isfile(imgPath)
            record = {}
            record['file_name'] = imgPath
            record['image_id'] = idx
            record['height'] = splitMask.shape[0]
            record['width']  = splitMask.shape[1]

            mask = splitMask
            cellNums = np.unique(mask)
            cellNums = cellNums[cellNums != 0]

            cells = []
            splitNum += 1
            for cellNum in cellNums:
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
                    "category_id": 0,
                }
                cells.append(cell)
            record["annotations"] = cells  
            datasetDicts.append(record)
            idx+=1
        return datasetDicts
              
# %%
if 'cellMorph_train' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph_train')
    print('Removing training')
if 'cellMorph_test' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph_test')
    print('Removing testing')
inputs = [experiment, imgType, 'train']

DatasetCatalog.register("cellMorph_" + "train", lambda x=inputs: getCells(inputs[0], inputs[1], inputs[2]))
MetadataCatalog.get("cellMorph_" + "train").set(thing_classes=["cell"])

DatasetCatalog.register("cellMorph_" + "test", lambda x=inputs: getCells(inputs[0], inputs[1], inputs[2]))
MetadataCatalog.get("cellMorph_" + "test").set(thing_classes=["cell"])


cell_metadata = MetadataCatalog.get("cellMorph_train")
# %%
cfg = get_cfg()
if not torch.cuda.is_available():
    print('CUDA not available, resorting to CPU')
    cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cellMorph_Train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# %%
cfg.OUTPUT_DIR = '../models/segmentation/mdamb436Seg'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
predictor = DefaultPredictor(cfg)
# %%
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, SemSegEvaluator
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator('cellMorph_test', output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "cellMorph_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# %% Get images not in train/test
imgName = '/home/user/work/cellMorph/data/TJ2406-MDAMB436-20X/raw/phaseContrast/phaseContrast_B2_2_2024y01m29d_10h25m.png'
imgName = '/home/user/work/cellMorph/data/TJ2406-MDAMB436-20X/raw/phaseContrast/phaseContrast_B2_1_2024y01m29d_10h25m.png'
img = imread(imgName)
imgsSplit = imageProcessing.imSplit(img)
imgSplit = imgsSplit[3]

plt.imshow(imgSplit, cmap = 'gray')

pcSplit = np.array([imgSplit, imgSplit, imgSplit]).transpose([1,2,0])
outputs = predictor(pcSplit)['instances'].to("cpu")
# %%
from src.visualization.segmentationVis import viewPredictorResult

viewPredictorResult(predictor, pcSplit)

# %%
imgSplit = imgsSplit[4]
pcSplit = np.array([imgSplit, imgSplit, imgSplit]).transpose([1,2,0])

plt.imshow(pcSplit)
outputs = predictor(pcSplit)['instances'].to("cpu")  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
nCells = len(outputs)
for cellNum in range(nCells):
    print(outputs[cellNum].scores)
    mask = outputs[cellNum].pred_masks.numpy()[0]
    pheno = 0

    contours = measure.find_contours(mask, .5)
    if len(contours) < 1:
        continue
    fullContour = np.vstack(contours)

    px = fullContour[:,1]
    py = fullContour[:,0]
    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
    poly = [p for x in poly for p in x]
    
    if outputs[cellNum].scores > .9:
        plt.plot(px, py)
# %%
viewPredictorResult(predictor, pcSplit)