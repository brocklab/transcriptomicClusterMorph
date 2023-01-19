#!/usr/bin/env python3
"""
This is a script that will automatically add new images to the correct folder

"""
# experimentFolder = 'TJ2201Split16'
# newFolder = 'TJ2201New'

# %%
import os
from cellMorphHelper import getImageBase, imSplit, printProgressBar
import sys
import cv2

def getImageBase(imName):
    """
    Gets the "base" information of an image. 
    Files should be named as follows:
    imageType_Well_WellSection_Date.extension

    The image base has no extension or image type

    Inputs:
    imName: String of image name

    Outputs:
    imageBase: The core information about the image's information in the incucyte
    """

    imageBase = '_'.join(imName.split('.')[0].split('_')[1:])

    return imageBase


# %% Validate images
newFolder = sys.argv[1]
experimentFolder = sys.argv[2]
print('Validating images...\n')
newFolder = os.path.join('../data', newFolder)
compositeFolder = os.path.join(newFolder, 'composite')
phaseContrastFolder = os.path.join(newFolder, 'phaseContrast')
compositeFiles = os.listdir(compositeFolder)
phaseContrastFiles = os.listdir(phaseContrastFolder)

assert len(compositeFiles) == len(phaseContrastFiles), 'Different # of files'

compositeBases, pcBases = [], []

# Check that bases line up
for composite, pc in zip(compositeFiles, phaseContrastFiles):
    compositeBase = getImageBase(composite)
    pcBase = getImageBase(pc)

    compositeBases.append(compositeBase)
    pcBases.append(pcBase)
for pc in pcBases:
    assert pc in compositeBases, 'Phase contrast image not in composite'
    compositeBases.remove(pc)

assert len(compositeBases) == 0, 'Non-matching composite and phase contrast images'


# %% Split images into existing directory
print('Splitting Images...\n')
nIms = 16
experimentFolder = os.path.join('../data', experimentFolder)
if not os.path.isdir(experimentFolder):
    os.makedirs(experimentFolder)
    print('Making new directory')

i = 1
printProgressBar(0, total = len(phaseContrastFiles), length = 50)
for pcName, compositeName in zip(phaseContrastFiles, compositeFiles):
    im = cv2.imread(os.path.join(phaseContrastFolder, pcName))
    tiles = imSplit(im)

    for num, im in enumerate(tiles):
        newImName =  '.'.join([pcName.split('.')[0]+'_'+str(num+1), pcName.split('.')[1]])
        newImPath = os.path.join(experimentFolder, 'phaseContrast', newImName)
        cv2.imwrite(newImPath, im)
    im = cv2.imread(os.path.join(compositeFolder, compositeName))
    tiles = imSplit(im)

    for num, im in enumerate(tiles):
        newImName =  '.'.join([compositeName.split('.')[0]+'_'+str(num+1), compositeName.split('.')[1]])
        newImPath = os.path.join(experimentFolder, 'composite', newImName)
        cv2.imwrite(newImPath, im)

    i+=1
    printProgressBar(i, total=len(phaseContrastFiles), length = 50)
