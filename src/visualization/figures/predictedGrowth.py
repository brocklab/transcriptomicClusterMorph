# %% [markdown]
"""
Residual plot of predicted vs actual growth
"""
# %%
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
# %%
homePath = Path('../../../')
datasetDictPath = homePath / './data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy'
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
# %%
wells = []
wells = [seg['file_name'].split('_')[1] for seg in datasetDicts]
uniqueWells, wellCts = np.unique(wells, return_counts=True)