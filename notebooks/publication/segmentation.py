# %%
from pathlib import Path
import numpy as np
from skimage.io import imread, imsave
from skimage import exposure
import matplotlib.pyplot as plt

from src.visualization.segmentationVis import viewPredictorResult
from src.models import modelTools
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
# %%
experiment = 'TJ2201'
# %%
predictor = modelTools.getSegmentModel('../../models/TJ2201Split16')
# %%
imNum = 2
imPath = Path(f'../../data/TJ2201/split16/phaseContrast/phaseContrast_E2_4_2022y04m07d_16h00m_{imNum}.png')
fullImPath = Path('../../data/TJ2201/raw/phaseContrast/phaseContrast_E2_4_2022y04m07d_16h00m.png')
img = imread(imPath)
imgFull = imread(fullImPath)
imgFull = exposure.equalize_adapthist(imgFull)
imsave('../../figures/publication/exemplar/segmentationFull.png', imgFull)
# How you would quickly visualize this with detectron2
# viewPredictorResult(predictor, imPath)
# %%
imgHighContrast = exposure.equalize_adapthist(img)
imsave('../../figures/publication/exemplar/patch.png', imgHighContrast)
outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
v = Visualizer(img[:, :, ::-1],
        #    metadata=cell_metadata, 
            scale=1, 
            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# Boxes
for box in outputs["instances"].pred_boxes.to('cpu'):
    v.draw_box(box)    
# Masks
for mask in outputs["instances"].pred_masks.to('cpu'):
    v.draw_soft_mask(mask)
v = v.get_output()
img =  v.get_image()[:, :, ::-1]
plt.imshow(img)
imsave('../../figures/publication/exemplar/segmentation.png', img)
# %%