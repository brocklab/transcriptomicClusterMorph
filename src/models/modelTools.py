from src.data.fileManagement import getImageBase

from skimage.io import imread
import matplotlib.pyplot as plt
import torch
import os
import time
from pathlib import Path
# from centermask.config import get_cfg

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

def getSegmentModel(modelPath: str, numClasses = 1):
    """
    Gets a segmentation model that can be used to output masks
    Inputs:
    modelPath: Folder with model. Final model must be named model_final.pth
    Outputs:
    Mask-RCNN model
    """
    # Insert segmentation folder to correctly load path
    modelPath = Path(modelPath)
    if modelPath.parts[-2] != 'segmentation':
        modelPathParts = list(modelPath.parts)
        modelPathParts.insert(-1, 'segmentation')
        modelPath = Path(*modelPathParts)
    modelPath = str(modelPath)
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
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = numClasses
    cfg.OUTPUT_DIR = modelPath
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    return predictor

def getLIVECell(confidenceThresh = 0.3, homePath = '..'):
    cfg = get_cfg()
    cfg.merge_from_file(f'{homePath}/data/sartorius/configs/bt474_config.yaml')
    cfg.MODEL.WEIGHTS = f"{homePath}/models/segmentation/LIVECell_anchor_free_model.pth"  # path to the model we just trained
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidenceThresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidenceThresh
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidenceThresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidenceThresh
    predictor = DefaultPredictor(cfg)
    return predictor

def printModelVariables(modelInputDict: dict):
    """
    Prints the names of variables used to train model. 

    This is recorded in the slurm .out file
    """
    modelDetailsPrint = '~ Model Details ~ \n'

    for var, inputVal in modelInputDict.items():
        modelDetailsPrint += f'{var} - {inputVal} \n'

    modelDetailsPrint += '-'*10 + '\n'
    print(modelDetailsPrint)
    return modelDetailsPrint

def getModelID(args):
    """
    Returns a model ID
    If a jobid is passed from a slurm file, the model ID is the job ID. 
    If there is no jobid passed, the model ID is an integer of the UNIX time.

    Inputs:
    - args: sys.argv from passing script

    Outputs:
    - modelID: ID used to name model
    - source: Source of modelID
    """

    modelID = int(time.time())
    source = 'time'

    return modelID, source
