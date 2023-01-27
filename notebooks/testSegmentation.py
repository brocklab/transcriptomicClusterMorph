# %%
from src.models import modelTools
from src.visualization import segmentationVis
# %%
predictor = modelTools.getSegmentModel('../models/TJ2201Split16')

# %%
dataPath = '../data/TJ2201/split16/phaseContrast/phaseContrast_C4_4_2022y04m06d_12h00m_14.png'

segmentationVis.viewPredictorResult(predictor, dataPath)

# %%
import numpy as np
datasetDicts = np.load('../data/TJ2201/split16/TJ2201DatasetDictCopy.npy', allow_pickle=True)
# %%
allWells = []
for imgSeg in datasetDicts:
    fileName = imgSeg['file_name']
    allWells.append(fileName.split('_')[1])


monoPos = ['B2','B3','B4','B5','B6','C2','C3','C4','C5','C6','D2','D3','D4','D5','D6']
monoNeg = ['E2','E3','E4','E5','E6','F2','F3','F4','F5','F6','G2','G3','G4','G5','G6']
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']

print(set(allWells))