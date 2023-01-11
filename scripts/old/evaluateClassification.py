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
# %matplotlib inline

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

# %%
from detectron2.structures import BoxMode
def getCellDicts(expDir, stage):
    
    labelDir = os.path.join(expDir, 'label', stage)

    imgs = os.listdir(labelDir)
    imgs = [img for img in imgs if img.endswith('.csv')]

    fluoro2Int = {'red': 0, 'green': 1}

    datasetDicts = []
    idx = 0
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

# %%
if "cellMorph_train" in DatasetCatalog:
    DatasetCatalog.remove("cellMorph_train")
    
expDir = '../data/TJ2201Split16'
stage = 'train'

inputs = [expDir, stage]
if "cellMorph_train" in DatasetCatalog:
    DatasetCatalog.remove("cellMorph_train")

DatasetCatalog.register("cellMorph_" + "train", lambda x=inputs: getCellDicts(inputs[0], inputs[1]))
MetadataCatalog.get("cellMorph_" + "train").set(thing_classes=["red", "green"])
cell_metadata = MetadataCatalog.get("cellMorph_train")

expDir = '../data/TJ2201Split16'
stage = 'val'

inputs = [expDir, stage]
if "cellMorph_val" in DatasetCatalog:
    DatasetCatalog.remove("cellMorph_val")

DatasetCatalog.register("cellMorph_" + "val", lambda x=inputs: getCellDicts(inputs[0], inputs[1]))
MetadataCatalog.get("cellMorph_" + "val").set(thing_classes=["red", "green"])
cell_metadata = MetadataCatalog.get("cellMorph_val")
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
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.OUTPUT_DIR = '../models/AG2021Classify'

# %%
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# %%
from detectron2.utils.visualizer import ColorMode

dataset_dicts = getCellDicts(inputs[0], inputs[1])
# %%
expDir = '../data/TJ2201Split16'
stage = 'val'

inputs = [expDir, stage]
random.seed(123)
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=cell_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(im)
    plt.subplot(122)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()
# %% Save training image that is predicted
expDir = '../data/AG2021Split16'
stage = 'train'

inputs = [expDir, stage]

dataset_dicts = getCellDicts(inputs[0], inputs[1])
# %%
random.seed(1234)
for d in random.sample(dataset_dicts, 4):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cell_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)

    plt.figure(figsize=(20,20))
    plt.subplot(121)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.title('Input Training Image')
    outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(img[:, :, ::-1],
                   metadata=cell_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.subplot(122)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.title('Phase Contrast Predicted Image')
    plt.savefig('../data/initMaskRCNNRes.png')


# %%
