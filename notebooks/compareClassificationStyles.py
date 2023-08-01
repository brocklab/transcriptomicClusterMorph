# %%
from pathlib import Path
import matplotlib.pyplot as plt

from src.visualization import trainTestRes
# %%
homePath = Path('..')
modelName = 'classifySingleCellCrop-1690581459'

resFileStamp = homePath / 'results' / 'classificationTraining' / f'{modelName}.txt'
x = trainTestRes.plotTrainingRes(resFileStamp)
# %%
modelName = 'classifySingleCellCrop-1690841803'
resFileBlackout = homePath / 'results' / 'classificationTraining' / f'{modelName}.txt'
x = trainTestRes.plotTrainingRes(resFileBlackout)
# %%
