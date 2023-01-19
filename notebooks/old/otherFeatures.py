# %% [markdown]

"""
# Cell Feature Extraction

This notebook is a testbed for extracting features from cells using pyfeats
"""
# %%
# import umap
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
import datetime
import pickle
import itertools
from cellMorph import cellPerims
# from cellMorphHelper import extractFeatures
import sys, importlib
# importlib.reload(sys.modules['cellMorphHelper'])
import cellMorphHelper

from skimage.color import rgb2gray
from skimage.measure import find_contours
from skimage.io import imread
import cv2

import pyfeats
# %%
cells = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-D2.pickle',"rb"))
cells = cellMorphHelper.filterCells(cells, confluencyDate=datetime.datetime(2022, 4, 8, 16, 0), color='green', edge=True)
# %% Pyfeatures test
# Take subset of cells
maxCells = 400
random.seed(1234)
cellSample = np.array(cells)[random.sample(range(len(cells)), 5000)]
cellSample = [cell for cell in cellSample if cell.color != 'NaN']
allFeatures, allLabels = [], []
y = []
c = 1
# %%
cell = cellSample[0]
colors = []
allFeatures, allLabels = [], []

cellNum = 1
for cell in cellSample[0:20]:
    f = rgb2gray(imread(cell.phaseContrast))
    mask = cell.mask
    a = np.where(mask==True)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    print(f'{cellNum}/{len(cellSample)}')
    f = f[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # Stretches values to be in between 0 and 255
    # NOTE: Image must initial be bound between 0 and 1
    f = f*255 
    mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(int)
    # Take only mask
    image = np.multiply(f, mask)
    perimeter = cell.perimeter
    # extractedFeatures, extractedLabels = cellMorphHelper.extractFeatures(image, mask, perimeter)

    # allFeatures.append(extractedFeatures)
    # allLabels.append(extractedLabels)
    try :
        extractedFeatures, extractedLabels = cellMorphHelper.extractFeatures(image, mask, perimeter)
        print(len(extractedFeatures))
        allFeatures.append(extractedFeatures)
        allLabels.append(extractedLabels)
        colors.append(cell.color)
    except:
        print('Couldn\'t get features for cell')
    if cellNum%100 == 0:
        print('Saving features')
        pickle.dump([allFeatures, allLabels], open('../results/allFeaturesTJ2201.pickle', "wb"))
    cellNum += 1
# %%
cell = cellSample[1]
colors = []
allFeatures, allLabels = [], []

cellNum = 1
cell = cellSample[1]
f = rgb2gray(imread(cell.phaseContrast))
mask = cell.mask
a = np.where(mask==True)
bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
f = f[bbox[0]:bbox[1], bbox[2]:bbox[3]]
# Stretches values to be in between 0 and 255
# NOTE: Image must initial be bound between 0 and 1
f = f*255 
mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(int)
# Take only mask
image = np.multiply(f, mask)
perimeter = cell.perimeter
# extractedFeatures, extractedLabels = cellMorphHelper.extractFeatures(image, mask, perimeter)

# allFeatures.append(extractedFeatures)
# allLabels.append(extractedLabels)
try :
    extractedFeatures, extractedLabels = cellMorphHelper.extractFeatures(image, mask, perimeter)

    allFeatures.append(extractedFeatures)
    allLabels.append(extractedLabels)
    colors.append(cell.color)
except Exception as e: print(e)

# %%
# %%
allLens = []
cellNum = 0
for cell in cellSample[0:20]:
    cellNum+=1
    lens = []

    f = rgb2gray(imread(cell.phaseContrast))
    mask = cell.mask
    a = np.where(mask==True)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    print(f'{cellNum}/{len(cellSample)}')
    f = f[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # Stretches values to be in between 0 and 255
    # NOTE: Image must initial be bound between 0 and 1
    f = f*255 
    mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(int)
    # Take only mask
    image = np.multiply(f, mask)
    perimeter = cell.perimeter
    perim = perimeter
    features = {}

    features['A_FOS']       = pyfeats.fos(image, mask)
    lens.append(len(features['A_FOS'][0]))
    features['A_GLCM']      = pyfeats.glcm_features(image, ignore_zeros=True)
    lens.append(len(features['A_GLCM'][0]))
    features['A_GLDS']      = pyfeats.glds_features(image, mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
    lens.append(len(features['A_GLDS'][0]))
    features['A_NGTDM']     = pyfeats.ngtdm_features(image, mask, d=1)
    lens.append(len(features['A_NGTDM'][0]))
    features['A_SFM']       = pyfeats.sfm_features(image, mask, Lr=4, Lc=4)
    lens.append(len(features['A_SFM'][0]))
    features['A_LTE']       = pyfeats.lte_measures(image, mask, l=7)
    lens.append(len(features['A_LTE'][0]))
    features['A_FDTA']      = pyfeats.fdta(image, mask, s=3)
    lens.append(len(features['A_FDTA'][0]))
    features['A_GLRLM']     = pyfeats.glrlm_features(image, mask, Ng=256)
    lens.append(len(features['A_GLRLM'][0]))
    features['A_FPS']       = pyfeats.fps(image, mask)
    lens.append(len(features['A_FPS'][0]))
    features['A_Shape_par'] = pyfeats.shape_parameters(image, mask, perim, pixels_per_mm2=1)
    lens.append(len(features['A_Shape_par'][0]))
    features['A_HOS']       = pyfeats.hos_features(image, th=[135,140])
    lens.append(len(features['A_HOS'][0]))
    features['A_LBP']       = pyfeats.lbp_features(image, image, P=[8,16,24], R=[1,2,3])
    lens.append(len(features['A_LBP'][0]))
    features['A_GLSZM']     = pyfeats.glszm_features(image, mask)
    lens.append(len(features['A_GLSZM'][0]))

    #% D. Multi-Scale features

    features['D_DWT'] =     pyfeats.dwt_features(image, mask, wavelet='bior3.3', levels=3)
    lens.append(len(features['D_DWT'][0]))
    features['D_SWT'] =     pyfeats.swt_features(image, mask, wavelet='bior3.3', levels=3)
    lens.append(len(features['D_SWT'][0]))
    # features['D_WP'] =      pyfeats.wp_features(image, mask, wavelet='coif1', maxlevel=3)
    # lens.append(len(features['D_WP'][0]))
    features['D_GT'] =      pyfeats.gt_features(image, mask)
    lens.append(len(features['D_GT'][0]))
    features['D_AMFM'] =    pyfeats.amfm_features(image)
    lens.append(len(features['D_AMFM'][0]))
    #% E. Other
    
    # features['E_HOG'] =             pyfeats.hog_features(image, ppc=8, cpb=3)
    # lens.append(len(features['E_HOG'][0]))
    features['E_HuMoments'] =       pyfeats.hu_moments(image)
    lens.append(len(features['E_HuMoments'][0]))
    features['E_ZernikesMoments'] = pyfeats.zernikes_moments(image, radius=9)
    lens.append(len(features['E_ZernikesMoments'][0]))
    #features['E_TAS'] =             pyfeats.tas_features(image)
    # Try to make a data frame out of it
    allFeatures, allLabels = [], []
    for label, featureLabel in features.items():

        if len(featureLabel) == 2:
            allFeatures += featureLabel[0].tolist()
            allLabels += featureLabel[1]
        else:
            assert len(featureLabel)%2 == 0
            nFeature = int(len(featureLabel)/2)

            allFeatures += list(itertools.chain.from_iterable(featureLabel[0:nFeature]))
            allLabels += list(itertools.chain.from_iterable(featureLabel[nFeature:]))
    allLens.append(lens)
# %%
