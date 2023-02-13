# %% [markdown]
"""
A notebook to compare morphological changes over time in coculture
"""
# %%
import pickle
import random
import pandas as pd
from src.data.cellMorphHelper import procrustes
import src.data.cellMorphHelper as cellMorphHelper
import numpy as np

import matplotlib.pyplot as plt
import umap
import datetime

# %%
def convertDate(date):
    """
    Returns a python datetime format of the Incucyte date format
    NOTE: This is very hardcoded and relies on a specific format. 

    Input example: 2022y04m11d_00h00m
    Output example: 2022-04-11 00:00:00
    """
    year =      int(date[0:4])
    month =     int(date[5:7])
    day =       int(date[8:10])
    hour =      int(date[12:14])
    minute =    int(date[15:17])

    date = datetime.datetime(year,month,day,hour,minute)

    return date
# %% Check dates

# Well E2 has alldates
esamNeg=pickle.load(open('../results/TJ2201Split16ESAMNeg.pickle',"rb"))
coculture=pickle.load(open('../results/TJ2201Split16CellPerims.pickle',"rb"))
# %%
e7Dates = []
e7Perims = []
e7Cells = []
for cell in coculture:
    imageBase = cell.imageBase
    well = imageBase.split('_')[0]
    date = '_'.join(imageBase.split('_')[2:4])
    if well == 'E7':
        e7Dates.append(date)
        e7Cells.append(cell)
e7Dates = [convertDate(date) for date in e7Dates]
# Align perimeters
e7Cells = cellMorphHelper.alignPerimeters(e7Cells)

for cell in e7Cells:
    e7Perims.append(cell.perimAligned.ravel())
# %% Reduce the size

nCells = 10000
random.seed(1234)
rPerm = random.sample(range(len(e7Perims)), len(e7Perims))[0:nCells]

e7PerimsSub = np.array(e7Perims)[rPerm]
e7DatesSub = np.array(e7Dates)[rPerm]

# %%
dateNums = np.array([int(date.strftime("%Y%m%d%H%M")) for date in e7DatesSub])
maxDate = max(dateNums)
minDate = min(dateNums)
dateNumsNorm = (dateNums-minDate)/(maxDate-minDate)
# Convert to percentages
# %% UMAP
fit = umap.UMAP()
u = fit.fit_transform(e7PerimsSub)
# %%
plt.scatter(u[:,0], u[:,1], s=2, c=np.exp(dateNumsNorm+100), cmap='plasma')
# plt.scatter(u[:,0], u[:,1], s=2, c=dateNumsNorm, cmap='plasma')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('ESAM Coculture UMAP Over Time')

# %%
import matplotlib.pyplot as plt
import numpy as np

# Generate data...
x = np.random.random(10)
y = np.random.random(10)

plt.scatter(x, y, c=y, s=500, cmap='Blues')
plt.show()
# %% Convert dates


