# %%
import cellMorph
import cellMorphHelper

import pickle
import os
import numpy as np
# %%
AG2217=pickle.load(open('../results/AG2217-ESAM-TGFB/AG2217-ESAM-TGFB.pickle',"rb"))
# %%
wells = []
for cell in AG2217:
    well = cell.imageBase.split('_')[0]
    wells.append(well)
# %%
wellDict = {well: [] for well in np.unique(wells)}

for cell in AG2217:
    well = cell.imageBase.split('_')[0]
    wellDict[well].append(cell)
# %%
saveDir = '../results/AG2217-ESAM-TGFB'
for well in wellDict.keys():
    saveFile = f'{saveDir}-{well}.pickle'
    print(saveFile)
    pickle.dump(wellDict[well], open(saveFile, "wb"))

# %%

# pickle.dump(foo, open(filename.pickle, "wb"))