# %%
from src.visualization.segmentationVis import  viewPredictorResult
from src.models import modelTools

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

import cv2
from skimage.io import imread
import matplotlib.pyplot as plt

from pathlib import Path
import random
import torch
import os

# %matplotlib inline
# %%
trainName = 'bt474Train'
testName = 'bt474Test'
if trainName in DatasetCatalog:
    DatasetCatalog.remove(trainName)
    MetadataCatalog.remove(trainName)
    print('Removing training')
if testName in DatasetCatalog:
    DatasetCatalog.remove(testName)
    MetadataCatalog.remove(testName)
    print('Removing testing')
register_coco_instances(trainName, {}, "../data/sartorius/segmentations/train.json", "../data/sartorius/images/livecell_train_val_images")
register_coco_instances(testName, {}, "../data/sartorius/segmentations/test.json", "../data/sartorius/images/livecell_test_images")
MetadataCatalog.get('bt474Train').set(thing_classes=["cell"])
MetadataCatalog.get('bt474Test').set(thing_classes=["cell"])
# %%
dataset_dicts_train = DatasetCatalog.get(trainName)
bt474_metadata = MetadataCatalog.get(trainName)
dataset_dicts_test = DatasetCatalog.get(testName)
# %%
numClasses = 1
modelPath = '../models/sartoriusBT474'
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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
# Inference
cfg.DETECTION_MAX_INSTANCES = 1000
cfg.POST_NMS_ROIS_INFERENCE = 8000
predictor = DefaultPredictor(cfg)
# %%
for d in random.sample(dataset_dicts_test, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :], metadata=bt474_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    plt.imshow(out.get_image()[:, :])
# %%
outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
v = Visualizer(img[:, :],
            #    metadata=cell_metadata, 
                scale=1, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.figure()
print("plotting")
plt.imshow(out.get_image()[:,:])
# plt.title(imBase)
plt.show()

# %%
