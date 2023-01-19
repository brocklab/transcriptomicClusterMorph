# %%
import sys
sys.path.append('../scripts')
import cellMorphHelper
from cellMorph import imgSegment

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
experiment = 'TJ2201'
segDir = os.path.join('../data',experiment,'segmentedIms')
segFiles = os.listdir(segDir)
segFiles = [segFile for segFile in segFiles if segFile.endswith('.npy')]

trainingEcc = []
for segFile in tqdm(segFiles):

    # Load in cellpose output
    segFull = os.path.join(segDir, segFile)
    seg = np.load(segFull, allow_pickle=True)
    seg = seg.item()

    mask = seg['masks']
    cellNums = np.unique(mask)
    cellNums = cellNums[cellNums != 0]

    for cellNum in cellNums:
        region = regionprops(mask.astype(np.uint8))
        if len(region)>1:
            region = sorted(region, key = lambda allprops: allprops.area)
        region = region[0]
        trainingEcc.append(region.eccentricity)
# %%
plt.violinplot([classEcc, segEcc, trainingEcc])
plt.xticks([1, 2, 3], ['Classification \n Model', 'Segmentation \n Model', 'Training Data'])
plt.ylabel('Eccentricity')
plt.show()