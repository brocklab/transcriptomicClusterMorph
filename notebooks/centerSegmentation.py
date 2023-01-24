# %%
import numpy as np

# %%

# %%
if torch.is_tensor(idx):
    idx = idx.tolist()

imgName = imgPaths[idx]
label = phenotypes[idx]
# %%
fullPath = os.path.join(dataPath, imgName)
maxRows, maxCols = maxImgSize, maxImgSize
img = imread(fullPath)

bb = bbs[idx]

nIncrease = nIncrease
colMin, rowMin, colMax, rowMax = bb
rowMin -= nIncrease
rowMax += nIncrease
colMin -= nIncrease
colMax += nIncrease

# Indexing checks
if rowMin <= 0:
    rowMin = 0
if rowMax > img.shape[0]:
    rowMax = img.shape[0]
if colMin <= 0:
    colMin = 0
if colMax >= img.shape[1]:
    colMax = img.shape[1]

# Increase the size of the bounding box and crop
bbIncrease = [colMin, rowMin, colMax, rowMax]
imgCrop = img[bbIncrease[1]:bbIncrease[3], bbIncrease[0]:bbIncrease[2]]

# Pad image
diffRows = int((maxRows - imgCrop.shape[0])/2)
diffCols = int((maxCols - imgCrop.shape[1])/2)
pcCrop = F.pad(torch.tensor(imgCrop[:,:,0]), pad=(diffCols, diffCols, diffRows, diffRows)).numpy()
pcCrop = resize(pcCrop, (maxRows, maxCols))

pcCrop = np.array([pcCrop, pcCrop, pcCrop]).transpose((1,2,0))
if transforms:
    img = transforms(pcCrop)