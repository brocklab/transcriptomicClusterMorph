# %%
# Convert image names
# TJ2201_D3_9_07d04h00m.tif - Bad
# phaseContrast_E2_9_2022y04m10d_04h00m.png - Good
import shutil
import os
from skimage.io import imread, imsave
import cellMorphHelper
from tqdm import tqdm
# %%
fileDir = '../data/TJ2201/phaseContrast/TJ2201-New'

files = os.listdir(fileDir)

for file in tqdm(files):

    fullPath = os.path.join(fileDir,file)

    components = file.split('.')
    name = components[0].split('_')

    well_imnum = name[1:3]
    date = name[-1]
    hourMin = date.split('d')[-1]
    dateNew = f'2022y04m{date[0:3]}_{hourMin}'

    imNameNew = '_'.join(['phaseContrast']+well_imnum+[dateNew])+'.png'
    img = imread(fullPath)
    newPath = os.path.join(fileDir, imNameNew)
    imsave(newPath, img)