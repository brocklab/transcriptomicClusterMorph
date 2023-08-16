# %%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.draw import polygon
from skimage.segmentation import clear_border
from skimage.morphology import binary_dilation

import detectron2.data.datasets as datasets
import detectron2
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
# %%
datasetDicts = datasets.load_coco_json(json_file='../../data/TJ2301-231C2/TJ2301-231C2Segmentations.json', image_root='')
datasetDicts = [record for record in datasetDicts if len(record['annotations']) > 0]
for record in tqdm(datasetDicts):
    for cell in record['annotations']:
        cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
        cell['bbox_mode'] = BoxMode.XYXY_ABS
# %%
breakOn = 0
ddNoBorder = []
for seg in tqdm(datasetDicts):
    segNoBorder = seg.copy()
    segNoBorder['annotations'] = []
    for i, annotation in enumerate(seg['annotations']):
        mask = np.zeros([seg['height'], seg['width']])
        polyx = annotation['segmentation'][0][::2]
        polyy = annotation['segmentation'][0][1::2]

        poly = np.array([[y,x] for x,y in zip(polyx, polyy)])
        rr, cc = polygon(poly[:, 0], poly[:, 1], mask.shape)

        mask[rr, cc] = [1]
        mask = binary_dilation(mask)

        maskCleared = clear_border(mask)

        if np.sum(maskCleared) != 0:
            segNoBorder['annotations'].append(annotation)
    ddNoBorder.append(segNoBorder)
    

nCellsInit, nCellsFilter = 0, 0
for segInit, segFilter in zip(datasetDicts, ddNoBorder):
    nCellsInit += len(segInit['annotations'])
    nCellsFilter += len(segFilter['annotations'])

print(f'Filtered from {nCellsInit} cells to {nCellsFilter}')
# np.save('../data/TJ2201/split16/TJ2201DatasetDictNoBorderFull.npy', ddNoBorder)
# %%
def getCells(datasetDict):
    return datasetDict
inputs = [ddNoBorder]
if 'cellMorph' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph')
    MetadataCatalog.remove('cellMorph')

DatasetCatalog.register("cellMorph", lambda x=inputs: getCells(inputs[0]))
MetadataCatalog.get("cellMorph").set(thing_classes=["cell"])

datasets.convert_to_coco_json('cellMorph', output_file='../../data/TJ2301-231C2/TJ2301-231C2SegmentationsNoBorder.json', allow_cached=False)

# %%
