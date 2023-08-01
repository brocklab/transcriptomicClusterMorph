# %%
import numpy as np
from pathlib import Path
from tqdm import tqdm

from skimage.draw import polygon_perimeter
from skimage.io import imread
# %%
experiment      = 'TJ2201'
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
# datasetDicts = [record for record in datasetDicts if len(record['annotations']) > 0]
# %%
imgPath = datasetDicts[0]['file_name'].split('/')[-1]
imgBase= '_'.join(imgPath.split('_')[0:-1])
fullImgPath = dataPath / f'{imgBase}.png'
# %%
datasetDictIms = {}
for entry in tqdm(datasetDicts):
    imgPath = entry['file_name'].split('/')[-1]
    imgBase= '_'.join(imgPath.split('_')[0:-1])
    if imgBase not in datasetDictIms.keys():
        datasetDictIms[imgBase] = []
    datasetDictIms[imgBase].append(entry)
# %%
imgBase = 'phaseContrast_D9_6_2022y04m05d_12h00m'

def createCellOutlines(datasetDictIms, imgBase):
    imgs = datasetDictIms[imgBase]
    if len(imgs) != 16:
        return np.zeros((1040, 1408))
    fullImg = {}
    for imgSeg in imgs:
        imgNum = int(imgSeg['file_name'].split('.png')[0].split('_')[-1])
        imgOutline = np.zeros((int(imgSeg['height']), int(imgSeg['width'])))
        for annotation in imgSeg['annotations']:
            seg = annotation['segmentation'][0]
            poly = np.reshape(seg, (int(len(seg)/2), 2))
            rr, cc = polygon_perimeter(poly[:,1], poly[:,0])
            imgOutline[rr,cc] = 1
        fullImg[imgNum] = imgOutline

    col1 = np.vstack([fullImg[1], fullImg[2], fullImg[3], fullImg[4]])
    col2 = np.vstack([fullImg[5], fullImg[6], fullImg[7], fullImg[8]])
    col3 = np.vstack([fullImg[9], fullImg[10], fullImg[11], fullImg[12]])
    col4 = np.vstack([fullImg[13], fullImg[14], fullImg[15], fullImg[16]])

    fullOutline = np.hstack([col1, col2, col3, col4])

    return fullOutline

imgOutlines = {}
for imgBase in tqdm(datasetDictIms.keys()):
    imgOutlines[imgBase] = createCellOutlines(datasetDictIms, imgBase)
# %%
imgs = datasetDictIms[imgBase]
if len(imgs) != 16:
    continue
    # return np.zeros((1040, 1408))
fullImg = {}
for imgSeg in imgs:
    imgNum = int(imgSeg['file_name'].split('.png')[0].split('_')[-1])
    imgOutline = np.zeros((int(imgSeg['height'])+1, int(imgSeg['width'])+1))
    for annotation in imgSeg['annotations']:
        seg = annotation['segmentation'][0]
        poly = np.reshape(seg, (int(len(seg)/2), 2))
        rr, cc = polygon_perimeter(poly[:,1], poly[:,0])
        imgOutline[rr,cc] = 1
    fullImg[imgNum] = imgOutline

col1 = np.vstack([fullImg[1], fullImg[2], fullImg[3], fullImg[4]])
col2 = np.vstack([fullImg[5], fullImg[6], fullImg[7], fullImg[8]])
col3 = np.vstack([fullImg[9], fullImg[10], fullImg[11], fullImg[12]])
col4 = np.vstack([fullImg[13], fullImg[14], fullImg[15], fullImg[16]])

fullOutline = np.hstack([col1, col2, col3, col4])

# %%
