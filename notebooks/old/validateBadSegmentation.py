# %%
from src.models.modelTools import getSegmentModel
from src.visualization.segmentationVis import viewPredictorResult

from skimage.io import imread
import matplotlib.pyplot as plt
# %%
model = getSegmentModel('../models/TJ2201Split16')
# %%
imPath = '../data/TJ2201/split16/phaseContrast/phaseContrast_C7_1_2022y04m07d_04h00m_1.png'
img = imread(imPath)

# %%
viewPredictorResult(model, imPath)