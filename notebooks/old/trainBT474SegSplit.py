# %%
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

import numpy as np
import cv2
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import json
from pathlib import Path
import random
import torch
import os
from tqdm import tqdm
# %matplotlib inline
# %%
homePath = Path('../../')
trainSegPath = homePath / "data/sartoriusSplit/segmentations/train.json"
f = open(trainSegPath) 

# returns JSON object as  
# a dictionary 
data = json.load(f) 
data["categories"] = [{"supercategory": "cell", "id": 1, "name": "cell"}]

with open(trainSegPath, 'w') as f:
    json.dump(data, f)
# %%
def preprocess(input_image, magnification_downsample_factor=1.0): 
    #internal variables
    #   median_radius_raw = used in the background illumination pattern estimation. 
    #       this radius should be larger than the radius of a single cell
    #   target_median = 128 -- LIVECell phase contrast images all center around a 128 intensity
    median_radius_raw = 75
    target_median = 128.0
    
    #large median filter kernel size is dependent on resize factor, and must also be odd
    median_radius = round(median_radius_raw*magnification_downsample_factor)
    if median_radius%2==0:
        median_radius=median_radius+1

    #scale so mean median image intensity is 128
    input_median = np.median(input_image)
    intensity_scale = target_median/input_median
    output_image = input_image.astype('float')*intensity_scale

    #define dimensions of downsampled image image
    dims = input_image.shape
    y = int(dims[0]*magnification_downsample_factor)
    x = int(dims[1]*magnification_downsample_factor)

    #apply resizing image to account for different magnifications
    output_image = cv2.resize(output_image, (x,y), interpolation = cv2.INTER_AREA)
    
    #clip here to regular 0-255 range to avoid any odd median filter results
    output_image[output_image > 255] = 255
    output_image[output_image < 0] = 0

    #estimate background illumination pattern using the large median filter
    background = cv2.medianBlur(output_image.astype('uint8'), median_radius)
    output_image = output_image.astype('float')/background.astype('float')*target_median

    #clipping for zernike phase halo artifacts
    output_image[output_image > 180] = 180
    output_image[output_image < 70] = 70
    output_image = output_image.astype('uint8')

    return output_image

# trainLoc = homePath / "data/sartoriusSplit/images/livecell_train_val_images"
# testLoc = homePath / "data/sartoriusSplit/images/livecell_test_images"

# trainFiles = list(trainLoc.iterdir())
# testFiles = list(testLoc.iterdir())
# for trainFile in tqdm(trainFiles):
#     img = imread(trainFile)
#     imgPreprocess = preprocess(img)

#     trainFileNew =  Path(str(trainFile).replace('/images/', '/processedImages/'))
#     imsave(trainFileNew, imgPreprocess)

# for testFile in tqdm(testFiles):
#     img = imread(testFile)
#     imgPreprocess = preprocess(img)

#     testFileNew =  Path(str(testFile).replace('/images/', '/processedImages/'))
#     imsave(testFileNew, imgPreprocess)
# %%
# categories = [{"supercategory": 1, "id": 1, "name": "cell"}]
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
register_coco_instances(trainName, {}, homePath / "data/sartoriusSplit/segmentations/train.json", homePath / "data/sartoriusSplit/processedImages/livecell_train_val_images")
register_coco_instances(testName, {},  homePath / "data/sartoriusSplit/segmentations/test.json",  homePath / "data/sartoriusSplit/processedImages/livecell_test_images")
MetadataCatalog.get('bt474Train').set(thing_classes=["cell"])
MetadataCatalog.get('bt474Test').set(thing_classes=["cell"])
# %%
dataset_dicts = DatasetCatalog.get(trainName)
bt474_metadata = MetadataCatalog.get(trainName)
# %%
# for d in random.sample(dataset_dicts, 1):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :], metadata=bt474_metadata, scale=2)
#     out = visualizer.draw_dataset_dict(d)
#     plt.imshow(out.get_image()[:, :])
# %%
cfg = get_cfg()
if not torch.cuda.is_available():
    print('CUDA not available, resorting to CPU')
    cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (trainName)
cfg.DATASETS.TEST = ()
print('L\n\n')
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations se .ems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cell). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = '../../models/segmentation/sartoriusBT474SplitPreprocessed'
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
cfg.INPUT.FORMAT = 'RGB'
predictor = DefaultPredictor(cfg)
# %%
from detectron2.evaluation import inference_on_dataset, SemSegEvaluator
from coco_evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator('bt474Train', output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "bt474Train")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# %%
