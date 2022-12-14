# %%
import numpy as np
from tqdm import tqdm
import os 

from skimage.io import imsave, imread
from skimage.draw import polygon2mask
from skimage.transform import rescale, resize

import torch
import torch.nn.functional as F
# %%
experiment = 'TJ2201'
# %% Read data
datasetDict = np.load(f'./{experiment}DatasetDict.npy', allow_pickle=True)
# %% Find details on class balance
phenoCount = {0: 0, 1: 0}
for img in datasetDict:
    for annotation in img['annotations']:
        phenoCount[annotation['category_id']] += 1

maxAmt = min(phenoCount.values())
# %% Scale down dataset dict so that there are equal annotation numbers
phenoCount = {0: 0, 1: 0}
balancedDatasetDict = []
for img in datasetDict:
    balancedDictImg = {}
    balancedDictImg['file_name'] = img['file_name']
    balancedDictImg['annotations'] = []
    for annotation in img['annotations']:
        cat = annotation['category_id']
        if phenoCount[cat] < maxAmt:
            balancedAnnotation = {}
            balancedAnnotation['bbox'] = annotation['bbox']
            balancedAnnotation['segmentation'] = annotation['segmentation']
            balancedAnnotation['category_id'] = annotation['category_id']
            balancedDictImg['annotations'].append(balancedAnnotation)
            phenoCount[cat] += 1
    balancedDatasetDict.append(balancedDictImg)
balancedDatasetDict = [img for img in balancedDatasetDict if len(img['annotations']) != 0]
# Validate
phenoCountBalanced = {0: 0, 1: 0}
for img in balancedDatasetDict:
    for annotation in img['annotations']:
        phenoCountBalanced[annotation['category_id']] += 1
phenoCountBalanced
# %%
def scaleAndCrop(img, polygon, bb, maxSize=150):
    """
    Returns a cropped image padded with black of maxSize

    Input:
        - img, 2d numpy array image
        - polygon, polygon from detectron2 dataset dicts
        - bb, bounding box surrounding segmented portion
        - maxSize, final size of cropped image (square)
    Output:
        - pcCrop, the cropped image as a 2d numpy array with black surrounding background
    """
    p1 = polygon[0::2]
    p2 = polygon[1::2]
    polygon = list(zip(p2, p1))
    mask = polygon2mask(img.shape[0:2], polygon)
    bb = [int(corner) for corner in bb]
    pcCrop = img[bb[1]:bb[3], bb[0]:bb[2]].copy()
    maskCrop = mask[bb[1]:bb[3], bb[0]:bb[2]].copy().astype('bool')

    pcCrop[~np.dstack((maskCrop,maskCrop,maskCrop))] = 0
    pcCrop = torch.tensor(pcCrop[:,:,0])

    # Keep aspect ratio and scale down data to be maxSize x maxSize (should be rare)
    maxRows, maxCols = maxSize, maxSize

    if pcCrop.shape[0]>maxRows:
        pcCrop = rescale(pcCrop, maxRows/pcCrop.shape[0])
    if pcCrop.shape[1]>maxCols:
        pcCrop = rescale(pcCrop, maxRows/pcCrop.shape[1])

    # Now pad out the amount to make it maxSize x maxSize
    diffRows = int((maxRows - pcCrop.shape[0])/2)+1
    diffCols = int((maxCols - pcCrop.shape[1])/2)
    pcCrop = F.pad(torch.tensor(pcCrop), pad=(diffCols, diffCols, diffRows, diffRows)).numpy()
    # Resize in case the difference was not actually an integer
    pcCrop = resize(pcCrop, (maxRows, maxCols))
    return pcCrop
# %%
label2Pheno = {0: 'esamPos', 1: 'esamNeg'}
savePath = f'../../data/{experiment}Split16SingleCell'
idx = 0
# Loop through segmentations
for imgData in tqdm(balancedDatasetDict, leave=False):
    # Read file once
    img = imread(imgData['file_name'])
    # Crop out each cell and save it
    for annotation in imgData['annotations']:
        bb = annotation['bbox']
        polygon = annotation['segmentation'][0]
        pcCrop = scaleAndCrop(img, polygon, bb)

        pheno = label2Pheno[annotation['category_id']]
        well = imgData['file_name'].split('_')[1]
        imgName = f'{pheno}_{well}_{idx}.jpg'
        fullPath = os.path.join(savePath, imgName)
        
        imsave(fullPath, pcCrop)
        idx += 1