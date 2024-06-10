# %%
from pathlib import Path
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

from src.visualization.segmentationVis import viewPredictorResult
from src.models import modelTools
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
# %%
experiment = 'TJ2201'
# %%
dataPath = Path(f'../../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
# %%
predictor = modelTools.getSegmentModel('../../models/TJ2201Split16')

# %%
imPathFull = Path(f'../../data/TJ2201/raw/phaseContrast/phaseContrast_E2_4_2022y04m07d_16h00m.png')
imDisp = imread(imPathFull)
factor = 200/322 #200 um/ 322 px
pxIncrease = int(200/factor)
imShape = imDisp.shape

imDisp[1000:1020, imShape[1]-pxIncrease-10:imShape[1]-10] = 0
plt.imshow(imDisp, cmap = 'gray')
imsave('../../figures/publication/exemplar/segmentationFull.png', imDisp)
# %%
imNum = 2
imPath = Path(f'../../data/TJ2201/split16//phaseContrast/phaseContrast_E2_4_2022y04m07d_16h00m_{imNum}.png')
im = imread(imPath)
imDisp = im[:,:,0]
factor = 200/322 #200 um/ 322 px
pxIncrease = int(50/factor)
imShape = im.shape

imDisp[240:250, imShape[1]-pxIncrease-10:imShape[1]-10] = 0
plt.imshow(imDisp, cmap = 'gray')
# How you would quickly visualize this with detectron2:
# viewPredictorResult(predictor, imPath)
# %%
# imBase = getImageBase(imPath.split('/')[-1])
outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
v = Visualizer(im[:, :, ::-1],
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
plt.axis('off')
plt.savefig('../../figures/publication/exemplar/segmentation.png', dpi = 500)
# %%
