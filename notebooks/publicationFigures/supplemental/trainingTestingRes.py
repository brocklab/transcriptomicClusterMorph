# %%
from pathlib import Path

from src.visualization.trainTestRes import plotTrainingRes
from src.data.fileManagement import getModelDetails
# %%
resName231 = 'classifySingleCellCrop-1707668614'
resName436 = 'classifySingleCellCrop-1715386736'
resNameTreat = 'classifySingleCellCrop-1715810868'

modelNames = {'MDA-MB-231 Subpopulations': resName231,
              'MDA-MB-436 Subpopulations': resName436,
              'Treated Populations': resNameTreat}
# %%
resFolder = Path('../../../results/classificationTraining')
for modelStr, name in modelNames.items():
    name += '.txt'
    resPath = resFolder / name

    plotTrainingRes(resPath, title=modelStr)