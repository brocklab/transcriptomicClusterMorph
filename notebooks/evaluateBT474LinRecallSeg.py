# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm

from detectron2.data.datasets import load_coco_json
import detectron2.data.datasets as datasets
from detectron2.data import MetadataCatalog, DatasetCatalog

from src.data import imageProcessing

# %%
# %%
def getGreenRecord(datasetDicts, datasetDictsGreen):
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []
        for annotation in record['annotations']:
            if annotation['category_id'] == 1:
                newAnnotations.append(annotation)
        if len(newAnnotations) > 0:
            record['annotations'] = newAnnotations
            datasetDictsGreen.append(record)
    return datasetDictsGreen

# %%
datasetDictsGreen = []

datasetDicts = load_coco_json('../data/TJ2442D/TJ2442DSegmentations.json', '.')
datasetDictsGreen = getGreenRecord(datasetDicts, datasetDictsGreen)
datasetDicts = load_coco_json('../data/TJ2442E/TJ2442ESegmentations.json', '.')
datasetDictsGreen = getGreenRecord(datasetDicts, datasetDictsGreen)
datasetDicts = load_coco_json('../data/TJ2442F/TJ2442FSegmentations.json', '.')
datasetDictsGreen = getGreenRecord(datasetDicts, datasetDictsGreen)

# %% Test to see if we found any very green cells
def getCells(datasetDict):
    return datasetDict
inputs = [datasetDictsGreen]
if 'cellMorph' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph')
    MetadataCatalog.remove('cellMorph')

DatasetCatalog.register("cellMorph", lambda x=inputs: getCells(inputs[0]))
MetadataCatalog.get("cellMorph").set(thing_classes=["cell"])

datasets.convert_to_coco_json('cellMorph', output_file='../data/misc/TJ24XXRecall.json', allow_cached=False)

# %%
for record in tqdm(datasetDicts):
    fileNameComposite = record['file_name'].replace('raw', 'split4').replace('phaseContrast', 'composite')
    img = imread(fileNameComposite)
    _, isAbbr = imageProcessing.removeImageAbberation(img)
    if isAbbr:
        print(fileNameComposite)
# %%
img = imread('../data/misc/loadThis_G6_3_00d00h00m.png')
img = img[:, :, 0:3]
_, isAbbr = imageProcessing.removeImageAbberation(img)

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
def plotComparison(im1, im2):
    plt.subplot(121)
    plt.imshow(im1)
    plt.subplot(122)
    plt.imshow(im2)


# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import morphology, measure

from src.data import imageProcessing

def dilN(im, n = 1):
    """Dilates image n number of times"""
    for i in range(n):
        im = morphology.binary_dilation(im)
    return im
def removeImageAbberation(RGB, thresh = 10000):
    """
    Block out very large areas where there are green spots in 
    fluorescence images. 
    
    Inputs: 
        - RGB: RGB image
        - thresh: Number of pixels required to intervene

    Outputs:
        - RGBNew: RGB image with aberration blocked out to median values
    """
    # Get BW image of very bright green objects
    nGreen, BW = imageProcessing.segmentGreenHigh(RGB)
    # Do a bit of processing to get an idea of where a cell might be
    # and where an abberation might be
    BW = morphology.remove_small_objects(BW)
    dil = morphology.binary_dilation(BW)
    # Find and remove blobs
    labels = measure.label(dil)
    unique, cts = np.unique(labels, return_counts=True)
    unique = unique[1:]
    cts = cts[1:]
    # Only take away very large aberrations, otherwise there's no solution likely
    numsHigh = unique[cts>thresh]
    if len(numsHigh) == 0:
        return RGB
    isAbberation = np.isin(labels, numsHigh)
    # Use convex hull to fully enclose cells
    convexAbberation = morphology.convex_hull_image(isAbberation)
    convexAbberation = dilN(convexAbberation, 50)

    RGBNew = img.copy()
    RGBNew[convexAbberation, 1] = np.median(RGBNew[:,:,1])
    RGBNew[convexAbberation, 2] = np.median(RGBNew[:,:,2])
    
    return RGBNew
img = imread('../data/misc/loadThis_G6_3_00d00h00m.png')
img = img[:, :, 0:3]
imgNew = removeImageAbberation(img)

plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.subplot(122)
plt.imshow(imgNew)
plt.axis('off')

plt.savefig('../figures/tempPres/imageAbberation.png', dpi = 500)
# %%
img = imread(fileNameComposite)
imgNew = removeImageAbberation(img)

