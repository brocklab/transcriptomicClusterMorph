# %%
import numpy as np
from skimage.io import imread, imsave
import os
import matplotlib.pyplot as plt
import pickle
# %%
maskDir = '../data/TJ2201/cellPoseOutputs'
saveFolder = '../data/TJ2201/masks'
maskFiles = [os.path.join(maskDir, file) for file in os.listdir(maskDir)]
mask = np.load(maskFiles[0], allow_pickle=True)
mask = mask.item()
imNum = 1
nIms = len(maskFiles)
for maskFile in maskFiles:
    print(f'{imNum}/{nIms}')
    fname = maskFile.split('.npy')[0]+'.tif'
    fname = fname.replace('composite','mask').replace('_seg','').replace(maskDir, saveFolder)
    mask = np.load(maskFile, allow_pickle=True).item()
    imsave(fname, mask['masks'])
    imNum += 1
# %%
# dict_keys(['img', 'outlines', 'masks', 'chan_choose', 'ismanual', 'filename', 'flows', 'est_diam'])
# %% Split mask directory
