# %%
import shutil
import os
import cellMorphHelper
import cellMorph
import pickle
import datetime
import numpy as np
# %%
coculture = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E7.pickle',"rb"))
# %%
ims = os.listdir('../data/TJ2201/composite')
imsTransfer = []
for im in ims:
    date = '_'.join(im.split('.')[0].split('_')[3:])
    date = cellMorphHelper.convertDate(date)
    well = im.split('_')[1]
    if date < datetime.datetime(2022, 4, 8, 16, 0) and well == 'E2':
        src = os.path.join('../data/TJ2201/composite', im)
        dest = os.path.join('/stor/scratch/Brock/Tyler/download', im)
        shutil.copy(src, dest)
# %%
newDir = '/stor/scratch/Brock/Tyler/download'