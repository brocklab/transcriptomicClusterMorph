# %%
# import umap
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np

from src.data.cellMorph import cellPerims
from skimage.color import rgb2gray
from skimage.measure import find_contours
from skimage.io import imread
import cv2

import pyfeats
import umap

# %%
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))
esamPos = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-D2.pickle',"rb"))

random.seed(1234)
esamNegPerm = random.sample(range(len(esamNeg)), len(esamNeg))
esamPosPerm = random.sample(range(len(esamPos)), len(esamPos))
nDesired = 400

esamNegSub = [esamNeg[x] for x in esamNegPerm if esamNeg[x].color=='red']
esamPosSub = [esamPos[x] for x in esamPosPerm if esamPos[x].color=='green']

esamNegSub = esamNegSub[0:nDesired]
esamPosSub = esamPosSub[0:nDesired]

cells = esamNegSub+esamPosSub
# %% Pyfeatures test
# Take subset of cells
# cellSample = np.array(cells)[random.sample(range(len(cells)), 5000)]
# cellSample = [cell for cell in cellSample if cell.color != 'NaN']
allFeatures, allLabels = [], []
y = []
c = 1

# Find textural features for each cell
for cell in cells:
    if cell.color == 'NaN':
        continue
    y.append(cell.color)
    print('Cell {}/{}'.format(c, len(cells)))
    f = rgb2gray(imread(cell.phaseContrast))
    mask = cell.mask
    a = np.where(mask==True)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])

    f = f[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    # Stretches values to be in between 0 and 255
    # NOTE: Image must initial be bound between 0 and 1
    image = f*255 
    mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(int)
    # Take only mask
    image = np.multiply(image, mask)

    perimeter = cell.perimeter

    # Write features and their meaning to list
    features, labels = [], []

    featuresFOS, labelsFOS = pyfeats.fos(image, mask)
    features+= list(featuresFOS)
    labels += labelsFOS

    features_mean, features_range, labels_mean, labels_range \
        =   pyfeats.glcm_features(image, ignore_zeros=True)
    features += list(features_mean)
    features += list(features_range)
    labels += labels_mean
    labels += labels_range

    featuresGLDS, labelsGLDS =   pyfeats.glds_features(image, mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
    features += list(featuresGLDS)
    labels += labelsGLDS

    featuresNGTDM, labelsNGTDM =  pyfeats.ngtdm_features(image, mask, d=1)
    features += list(featuresNGTDM)
    labels += labelsNGTDM

    featuresSFM, labelsSFM =    pyfeats.sfm_features(image, mask, Lr=4, Lc=4)
    features += list(featuresSFM)
    labels += labelsSFM

    featuresLTE, labelsLTE =    pyfeats.lte_measures(image, mask, l=7)
    features += list(featuresLTE)
    labels += labelsLTE

    featuresFDTA, labelsFDTA =   pyfeats.fdta(image, mask, s=3)
    features += list(featuresFDTA)
    labels += labelsFDTA

    featuresGLRLM, labelsGLRLM =  pyfeats.glrlm_features(image, mask, Ng=256)
    features += list(featuresGLRLM)
    labels += labelsGLRLM

    featuresFPS, labelsFPS =    pyfeats.fps(image, mask)
    features += list(featuresFPS)
    labels += labelsFPS

    featuresShape, labelsShape = pyfeats.shape_parameters(image, mask, perimeter, pixels_per_mm2=1)
    features += list(featuresShape)
    labels += labelsShape

    featuresHOS, labelsHOS =     pyfeats.hos_features(image, th=[135,140])
    features += list(featuresHOS)
    labels += labelsHOS

    featuresLBP, labelsLBP =     pyfeats.lbp_features(image, image, P=[8,16,24], R=[1,2,3])
    features += list(featuresLBP)
    labels += labelsLBP

    featuresGLSZM, labelsGLSZM =   pyfeats.glszm_features(image, mask)
    features += list(featuresGLSZM)
    labels += labelsGLSZM

    allFeatures.append(features)
    allLabels.append(labels)
    c += 1
# %%
X = pd.DataFrame(allFeatures)
X['label'] = y
X = X.dropna()

yNA = X['label']
X = X.drop('label', axis=1)
# %%
fit = umap.UMAP()
u = fit.fit_transform(X)
# %%
plt.scatter(u[:,0], u[:,1],c=yNA, s = 5, alpha = 0.25)
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, yNA, test_size=0.33, random_state=1234)
clf = LogisticRegression(solver="liblinear", random_state=1234, C=1e-6,max_iter=1e7).fit(X_train, y_train)
roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
# %%
