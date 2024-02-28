# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from detectron2.data.datasets import load_coco_json
from src.models.
# %%
datasetDicts = load_coco_json('../data/TJ2442D/TJ2442DSegmentations.json', '.')
# %% Test to see if we found any very green cells
datasetDictsGreen = []
for record in datasetDicts:
    record = record.copy()
    newAnnotations = []
    for annotation in record['annotations']:
        if annotation['category_id'] == 1:
            newAnnotations.append(annotation)
    if len(newAnnotations) > 0:
        record['annotations'] = newAnnotations
        datasetDictsGreen.append(record)



# %%
record = datasetDictsGreen[15]
fileNameComposite = record['file_name'].replace('raw', 'split4').replace('phaseContrast', 'composite')
fileNamePhaseContrast = record['file_name'].replace('raw', 'split4').replace('phaseContrast', 'composite')

imgComposite = imread(fileNameComposite)
annotations = record['annotations']
annotation = annotations[0]
poly = annotation['segmentation'][0]
polyx = poly[::2]
polyy = poly[1::2]

plt.imshow(imgComposite)
plt.plot(polyx, polyy, 'r')
plt.title(len(annotations))
# print(fileNameComposite)
os.system(f'xdg-open {fileNameComposite}')

# %%
from src.data.imageProcessing import imSplit, segmentGreenHigh

nGreen, BW = segmentGreenHigh(imgComposite)