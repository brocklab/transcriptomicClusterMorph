# %%
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

from skimage.draw import polygon2mask
from skimage import morphology
from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode
# %%
def getGreenRecord(datasetDicts, datasetDictsGreen = []):
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []

        for annotation in record['annotations']:
            annotation['bbox'] = BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            if annotation['category_id'] == 1:
                newAnnotations.append(annotation)
        if len(newAnnotations) > 0:
            record['annotations'] = newAnnotations
            datasetDictsGreen.append(record)
    return datasetDictsGreen
# %% Get green uncalibrated for sorted cells
datasetDictsSort = np.load('../data/TJ2443LineageRecallSort/TJ2443LineageRecallSortDatasetDicts.npy', allow_pickle=True)
# %%
idx = 1
record = datasetDictsSort[idx]
fileName = record['file_name'].replace('raw', 'split4')

greenImg = imread(fileName.replace('phaseContrast', 'greenCalibrated'))
compositeImg = imread(fileName.replace('phaseContrast', 'composite'))

strel = morphology.disk(1)
# compositeImg = compositeImg[:,:,1]
# compositeImg = morphology.white_tophat(compositeImg[:,:,1], strel)

imgShape = greenImg.shape
plt.figure(figsize=(20,15))
plt.subplot(121)
plt.imshow(greenImg)
plt.axis('off')
c = 0
fullMask = np.zeros(imgShape)
for annotation in record['annotations']:
    seg = annotation['segmentation'][0]
    polyx = seg[::2]
    polyy = seg[1::2]

    plt.plot(polyx, polyy)

    mask = polygon2mask(imgShape, list(zip(polyy, polyx)))

    fullMask += mask
    greenVals = mask*greenImg
    greenVals = greenVals[greenVals>0].ravel()
    greenMetric = np.round(np.median(greenVals))
    plt.annotate(greenMetric, xy = (polyx[0], polyy[0]), c = 'white')


    c += 1

plt.subplot(122)
plt.imshow(compositeImg)

fullMask[fullMask == 0] = 100
fullMask[fullMask < 100] = 0
fullMask[fullMask == 100] = 1

greenBackground = greenImg * fullMask

meanBackground = np.mean(greenBackground[greenBackground > 0])
medianBackground = np.median(greenBackground[greenBackground > 0])

print(f'Mean: {meanBackground:0.2f} \t Median: {medianBackground:0.2f}')
# %%

img = imread('/home/user/work/cellMorph/data/TJ2443LineageRecallSort/raw/greenCalibrated/greenCalibrated_B2_1_2024y03m05d_10h45m.tif')

# %%
plt.imshow(img)