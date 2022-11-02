# %%
import sys
sys.path.append('../scripts')
import cellMorphHelper
from cellMorph import imgSegment

import pickle
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import regionprops
from skimage.io import imread
# %%
segs = pickle.load(open('../data/fullSegmentationTJ2201Test.pickle',"rb"))
classes = pickle.load(open('../data/fullClassificationAccuracy.pickle',"rb"))
# %% Convert to easier format
segDict, classesDict = {}, {}
for seg in segs:
    fname = seg.pcImg.split('/')[-1]
    segDict[fname] = seg
for classification in classes:
    fname = classification.pcImg.split('/')[-1]
    classesDict[fname] = classification
# %%
nSeg, nClass = 0, 0
segEcc, classEcc = [], []
for img in segDict.keys():
    seg = segDict[fname]
    classification = classesDict[fname]

    for segMask in seg.masks:
        region = regionprops(segMask.astype(np.uint8))
        if len(region)>1:
            print('hi')
            region = sorted(region, key = lambda allprops: allprops.area)
        region = region[0]
        segEcc.append(region.eccentricity)

    for classificationMask in classification.masks:
        region = regionprops(classificationMask.astype(np.uint8))
        if len(region)>1:
            region = sorted(region, key = lambda allprops: allprops.area)
        region = region[0]
        classEcc.append(region.eccentricity)
# %%
plt.violinplot([classEcc, segEcc])
plt.xticks([1, 2], ['Classification \n Model', 'Segmentation \n Model'])
plt.ylabel('Eccentricity')
plt.show()
# %%
fnames = list(segDict.keys())

n = 603
segDict[fnames[n]].imshow()
plt.show()
classesDict[fnames[n]].imshow()

# %%