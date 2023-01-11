# %% [markdown]
"""
# Cell Segmentation With Mask-RCNN
"""
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
import os, json, cv2, random
import matplotlib.pyplot as plt

# Import image processing
from skimage import measure
from skimage import img_as_float
from scipy.spatial import ConvexHull

# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
# %% Data loader
def getCellDicts(expDir, stage):
    labelDir = os.path.join(expDir, 'label', stage)

    imgs = os.listdir(labelDir)
    imgs = [img for img in imgs if img.endswith('.csv')]

    datasetDicts = []
    idx = 0
    fluoro2Int = {'red': 0, 'green': 1}

    for img in imgs:

        # Information for the whole image
        imgBase = '_'.join(img.split('_')[1:])[:-4]
        
        annos = pd.read_csv(os.path.join(labelDir, img))
        for splitNum in range(1,5):
            phaseContrastName = 'phaseContrast_'+imgBase+'_'+str(splitNum)+'.jpg'
            phaseContrastPath = os.path.join(expDir, 'phaseContrast',phaseContrastName)
            height, width = cv2.imread(phaseContrastPath).shape[:2]

            record = {}
            record['file_name'] = phaseContrastPath
            record['image_id'] = idx
            record['height'] = height
            record['width'] = width
            # Cell information is stored in a .csv
            # Load the corresponding image, and store its information
            maskName = 'mask_'+imgBase+'_'+str(splitNum)+'.tif'
            maskPath = os.path.join(expDir, 'mask', maskName)
            imgMask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
            objs = []

            for maskLabel, fluorescence in zip(annos['maskLabel'], annos['fluorescence']):
                # Contour converts the mask to a polygon
                contours = measure.find_contours(img_as_float(imgMask==maskLabel), .5)
                # The convex hull is used to merge any extra contours
                hull = []
                for contour in contours:
                    hull+=contour[ConvexHull(contour).vertices].tolist()
                # If a cell is found, it's added to the list
                if len(hull)>0:
                    hull = np.array(hull)[ConvexHull(hull).vertices]

                    px = hull[:,1]
                    py = hull[:,0]
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]

                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": fluoro2Int[fluorescence],
                    }
                    objs.append(obj)
            record["annotations"] = objs
            datasetDicts.append(record)
    return datasetDicts

expDir = '../data/AG2021Split16'
stage = 'train'

inputs = [expDir, stage]
if "cellMorph_train" in DatasetCatalog:
    DatasetCatalog.remove("cellMorph_train")

DatasetCatalog.register("cellMorph_" + "train", lambda x=inputs: getCellDicts(inputs[0], inputs[1]))
MetadataCatalog.get("cellMorph_" + "train").set(thing_classes=["red", "green"])
cell_metadata = MetadataCatalog.get("cellMorph_train")
# %% Visualization of data loader
datasetDicts = getCellDicts(inputs[0], inputs[1])
for d in random.sample(datasetDicts, 3):
    print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cell_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()
# %%
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
cfg.OUTPUT_DIR = '../models/AG2021Classify'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()



# %%
