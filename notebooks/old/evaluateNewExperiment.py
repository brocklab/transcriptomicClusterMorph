# %%
import sys, importlib
# importlib.reload(sys.modules['cellMorphHelper'])

from cellMorphHelper import splitExpIms, getSegmentModel, viewPredictorResult
import cellMorphHelper
import random

from skimage.io import imread

import matplotlib.pyplot as plt
# %%
predictor = cellMorphHelper.getSegmentModel('../models/AG2021Split16')
# %% Split Experiment Images
experiment = 'TJ2201'
nIms = 16
# splitExpIms(experiment, nIms)
# %% View segmentations
pcPath = os.path.join('../data/'+experiment+'Split'+str(nIms), 'phaseContrast')
ims = os.listdir(pcPath)

random.seed(1234)
for imNum in random.sample(range(len(ims)), 3):
    imPath = os.path.join(pcPath, ims[imNum])
    cellMorphHelper.viewPredictorResult(predictor, imPath)
# %%
cellMorphHelper.viewPredictorResult(predictor, '../data/TJ2201Split16/phaseContrast/phaseContrast_D2_1_2022y04m09d_04h00m_14.png')