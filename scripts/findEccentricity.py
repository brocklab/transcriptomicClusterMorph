# %% 
import pickle
import cellMorphHelper
import cellMorph
import datetime
import numpy as np

from skimage.measure import label, regionprops, regionprops_table
from skimage.io import imread
# %%
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))
esamPos = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-D2.pickle',"rb"))
# %%
esamNeg = cellMorphHelper.filterCells(esamNeg, confluencyDate=datetime.datetime(2022, 4, 8, 16, 0), color='red', edge=True)
esamPos = cellMorphHelper.filterCells(esamPos, confluencyDate=datetime.datetime(2022, 4, 8, 16, 0), color='green', edge=True)

cells = esamNeg+esamPos
ecc = []
c = 0
for cell in cells:
    region = regionprops(cell.mask.astype(np.uint8))
    if len(region)>1:
        print(c)
        region = sorted(region, key = lambda allprops: allprops.area)
        break
    region = region[0]
    ecc.append(region.eccentricity)
    c+=1
    if c%10000 == 0:
        print(f'{c}/{len(cells)}')
# %%
plt.hist(ecc)
plt.title('Eccentricity of Cells')
plt.xlabel('Eccentricity (0 == Circle)')
plt.ylabel('Number of Cells')
# %%
circularCells = np.where(np.array(ecc)<0.4)[0]
print(len(circularCells))
cells[circularCells[10]].imshow()
