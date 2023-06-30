# %%
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
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
homePath = Path('../../')
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
register_coco_instances(trainName, {}, homePath / "data/sartorius/segmentations/train.json", homePath / "data/sartorius/images/livecell_train_val_images")
register_coco_instances(testName, {},  homePath / "data/sartorius/segmentations/test.json",  homePath / "data/sartorius/images/livecell_test_images")
MetadataCatalog.get('bt474Train').set(thing_classes=["cell"])
MetadataCatalog.get('bt474Test').set(thing_classes=["cell"])
# %%
dataset_dicts = DatasetCatalog.get(trainName)
bt474_metadata = MetadataCatalog.get(trainName)
# %%
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :], metadata=bt474_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    plt.imshow(out.get_image()[:, :])
# %%
cfg = get_cfg()
if not torch.cuda.is_available():
    print('CUDA not available, resorting to CPU')
    cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (trainName)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations se .ems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cell). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = '../../models/segmentation/sartoriusBT474'
# Recommended from https://github.com/matterport/Mask_RCNN/issues/1884
cfg.RPN_TRAIN_ANCHORS_PER_IMAGE = 800
cfg.MAX_GT_INSTANCES = 300
cfg.PRE_NMS_LIMIT = 12000
cfg.POST_NMS_ROIS_TRAINING = 6000
# Inference
cfg.DETECTION_MAX_INSTANCES = 1000
cfg.POST_NMS_ROIS_INFERENCE = 8000
# %%
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# %%
from detectron2.engine import DefaultPredictor
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
predictor = DefaultPredictor(cfg)
# %%
from detectron2.evaluation import inference_on_dataset, SemSegEvaluator
from coco_evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
# %%
evaluator = COCOEvaluator('bt474Train', output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "bt474Train")
print(inference_on_dataset(predictor.model, val_loader, evaluator))