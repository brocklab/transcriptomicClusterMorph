from src.data.fileManagement import getImageBase

from skimage.io import imread
import matplotlib.pyplot as plt
import torch
import os

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


def printModelVariables(modelInputDict: dict):
    """
    Prints the names of variables used to train model. 

    This is recorded in the slurm .out file
    """
    print('----')

    for var, inputVal in modelInputDict.items():
        print(f'{var} - {inputVal}')

    print('----')