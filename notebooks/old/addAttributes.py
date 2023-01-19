# %% [markdown]
"""
This is a notebook for adding attributes to the cellMorph class
"""
# %%
import pickle
import cellMorph
import cellMorphHelper
# %%
file = '../results/AG2217-ESAM-TGFB/AG2217-ESAM-TGFB.pickle'
cells=pickle.load(open(file,"rb"))
# coculture=pickle.load(open('../results/TJ2201Split16CellPerims.pickle',"rb"))
# %%
# Check well
cell = esamNeg[0]
imageBase = cell.imageBase
well = imageBase.split('_')[0]
# Check date
date = imageBase.split('_')[2:-1]
assert 'y' in date[0], 'No year in date, check to make sure it is a split image'
date = '_'.join(date)
date = cellMorphHelper.convertDate(date)
print(date)
# %% Convert and add
for cell in cells:
    well = imageBase.split('_')[0]
    date = imageBase.split('_')[2:-1]
    assert 'y' in date[0], 'No year in date, check to make sure it is a split image'
    date = '_'.join(date)
    date = cellMorphHelper.convertDate(date)

    cell.well = well
    cell.date = date
pickle.dump(cells, open('../results/TJ2201Split16ESAMNeg.pickle', "wb"))