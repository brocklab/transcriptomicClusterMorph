# %% 
"""
cellMorphHelper.py contains various helper functions to aid analysis, such as setting up directories and splitting images. 

The functions can be split into roughly 3 categories:
1. Helper functions for file management
2. Helper functions for image processing/analysis
3. Helper functions for more easily calling Detectron2 utilities such as visualization and classification
"""
# %%
# import some common libraries
import shutil
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import detectron2
import datetime
import random
import itertools
from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

# Import image processing
from skimage import measure
from skimage import img_as_float
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border
import pyfeats

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# %%
# File Management

def makeNewExperimentDirectory(experimentName):
    """
    Properly populates a new blank experiment data directory
    Inputs:
    - experimentName: Name of new experiment
    Outputs:
    None
    """
    assert os.path.isdir('../data')

    # Make base directory for experiment
    dataFolder = os.path.join('../data', experimentName)

    os.makedirs(dataFolder, exist_ok=True)

    # Make remaining sub-folders
    composite = os.path.join('../data', experimentName, 'composite')
    labelTrain = os.path.join('../data', experimentName, 'label', 'train')
    labelVal = os.path.join('../data', experimentName, 'label', 'val')
    mask = os.path.join('../data', experimentName, 'mask')
    phaseContrast = os.path.join('../data', experimentName, 'phaseContrast')

    newFolders = [composite, labelTrain, labelVal, mask, phaseContrast]

    for folder in newFolders:
        os.makedirs(folder, exist_ok=True)

def imSplit(im, nIms: int=16):
    """
    Splits images into given number of tiles
    Inputs:
    im: Image to be split
    nIms: Number of images (must be a perfect square)

    Outputs:
    List of split images
    """
    div = int(np.sqrt(nIms))

    nRow = im.shape[0]
    nCol = im.shape[1]

    M = nRow//div
    N = nCol//div
    tiles = []
    for y in range(0,im.shape[1],N): # Column
        for x in range(0,im.shape[0],M): # Row
            tiles.append(im[x:x+M,y:y+N])
    return tiles

def splitExpIms(experiment, nIms=16):
    """
    Splits up Incucyte experiment into given number of chunks. Does this for both masks
    and phase contrast images
    Inputs:
    experiment: Experiment name located under ../data/
    nIms: Number of tiles for each image
    Outputs:
    New and populated directory
    """

    # Split masks
    dataDir = os.path.join('../data',experiment)
    splitDir = os.path.join('../data',experiment+'Split'+str(nIms))

    # Make new directory
    if os.path.isdir(splitDir):
        shutil.rmtree(splitDir)
    
    makeNewExperimentDirectory(splitDir)

    print('Splitting masks')    
    # Split and save masks
    maskDir = os.path.join(dataDir, 'mask')
    if os.path.isdir(maskDir):
        maskNames = os.listdir(maskDir)

        for maskName in maskNames:
            # Read and split mask
            mask = cv2.imread(os.path.join(dataDir, 'mask', maskName), cv2.IMREAD_UNCHANGED)
            maskSplit = imSplit(mask, nIms)

            # For each mask append a number, then save it
            for num, mask in enumerate(maskSplit):
                newMaskName =  '.'.join([maskName.split('.')[0]+'_'+str(num+1), maskName.split('.')[1]])
                newMaskPath = os.path.join(splitDir, 'mask', newMaskName)
                cv2.imwrite(newMaskPath, mask)

    print('Splitting images')
    # Split up phase contrast images
    pcDir = os.path.join(dataDir, 'phaseContrast')
    if os.path.isdir(pcDir):
        imNames = os.listdir(pcDir)

        for imName in tqdm(imNames):
            # Read and split mask
            im = cv2.imread(os.path.join(dataDir, 'phaseContrast', imName))
            tiles = imSplit(im, nIms)
            # For each mask append a number, then save it
            for num, im in enumerate(tiles):
                newImName =  '.'.join([imName.split('.')[0]+'_'+str(num+1), imName.split('.')[1]])
                newImPath = os.path.join(splitDir, 'phaseContrast', newImName)
                cv2.imwrite(newImPath, im)

    print('Splitting composite')
    # Split up composite images
    compositeDir = os.path.join(dataDir, 'composite')
    if os.path.isdir(compositeDir):
        imNames = os.listdir(compositeDir)

        for imName in tqdm(imNames):
            # Read and split mask
            im = cv2.imread(os.path.join(dataDir, 'composite', imName))
            tiles = imSplit(im, nIms)
            # For each mask append a number, then save it
            for num, im in enumerate(tiles):
                newImName =  '.'.join([imName.split('.')[0]+'_'+str(num+1), imName.split('.')[1]])
                newImPath = os.path.join(splitDir, 'composite', newImName)
                cv2.imwrite(newImPath, im)   
    # Copy over labels
    originalTrain = os.path.join(dataDir, 'label', 'train')
    originalVal = os.path.join(dataDir,'label', 'val')

    if os.path.isdir(originalTrain):
        newTrain = os.path.join(splitDir, 'label', 'train')
        newVal =   os.path.join(splitDir,'label', 'val')

        shutil.rmtree(newTrain)
        shutil.rmtree(newVal)
        
        shutil.copytree(originalTrain, newTrain)
        shutil.copytree(originalVal, newVal)

def convertLabels(labelDir):
    """
    Converts labels from their .csv representation to a pickled dictionary
    Inputs:
    labelDir: Directory where labels are stored as train and val
    Outputs:
    A nested pickled dictionary for each image containing
    information about each cell's identity
    """
    labels = {}
    # Walk through directory and add each image's information
    for root, dirs, files in os.walk(labelDir):
        print(root)
        for imageLabel in files:
            if imageLabel.endswith('.csv'):
                labelDf = pd.read_csv(os.path.join(root,imageLabel))
                imBase = '_'.join(imageLabel.split('.')[0].split('_')[1:])

                maskLabels = labelDf['maskLabel']
                groups = labelDf['fluorescence']

                # Each image also has a dictionary which is accessed by the mask label
                labels[imBase] = {}
                for maskLabel, group in zip(maskLabels, groups):
                    labels[imBase][maskLabel] = group
    saveName = os.path.join(labelDir, 'labels.pkl')
    pickle.dump(labels, open(saveName, "wb"))

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

def convertDate(date):
    """
    Returns a python datetime format of the Incucyte date format
    NOTE: This is very hardcoded and relies on a specific format. 

    Input example: 2022y04m11d_00h00m
    Output example: 2022-04-11 00:00:00
    """
    year =      int(date[0:4])
    month =     int(date[5:7])
    day =       int(date[8:10])
    hour =      int(date[12:14])
    minute =    int(date[15:17])

    date = datetime.datetime(year,month,day,hour,minute)

    return date

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)

    Credit: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters?noredirect=1&lq=1
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
# Image processing 
def clearEdgeCells(cell):
    """
    Checks if cells are on border by dilating them and then clearing the border. 
    NOTE: This could be problematic since some cells are just close enough, but could be solved by stitching each image together, then checking the border.
    """
    mask = cell.mask
    maskDilate = binary_dilation(mask)
    maskFinal = clear_border(maskDilate)
    if np.sum(maskFinal)==0:
        return 0
    else:
        return 1

def segmentGreen(RGB):
    """
    Finds green pixels from Incucyte data
    Input: RGB image
    Output: # of green pixels and mask of green pixels
    """
    # def segment
    I = rgb2hsv(RGB)

    # Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.129
    channel1Max = 0.845

    # Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.309
    channel2Max = 1.000

    # Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.761
    channel3Max = 1.000

    # Create mask based on chosen histogram thresholds
    sliderBW =  np.array(I[:,:,0] >= channel1Min ) & np.array(I[:,:,0] <= channel1Max) & \
                np.array(I[:,:,1] >= channel2Min ) & np.array(I[:,:,1] <= channel2Max) & \
                np.array(I[:,:,2] >= channel3Min ) & np.array(I[:,:,2] <= channel3Max)
    BW = sliderBW

    # Initialize output masked image based on input image.
    maskedRGBImage = RGB.copy()

    # Set background pixels where BW is false to zero.
    maskedRGBImage[~np.dstack((BW, BW, BW))] = 0

    nGreen = np.sum(BW)
    return nGreen, BW

def segmentRed(RGB):
    """
    Finds red pixels from Incucyte data
    Input: RGB image
    Output: # of red pixels and mask of green pixels
    """
    # Convert RGB image to chosen color space
    I = rgb2hsv(RGB)

    # Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.724
    channel1Max = 0.185

    # Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.277
    channel2Max = 1.000

    # Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.638
    channel3Max = 1.000

    # Create mask based on chosen histogram thresholds
    sliderBW =  np.array(I[:,:,0] >= channel1Min )  | np.array(I[:,:,0] <= channel1Max)  & \
                np.array(I[:,:,1] >= channel2Min ) &  np.array(I[:,:,1] <= channel2Max) & \
                np.array(I[:,:,2] >= channel3Min ) &  np.array(I[:,:,2] <= channel3Max)
    BW = sliderBW

    # Initialize output masked image based on input image.
    maskedRGBImage = RGB.copy()

    # Set background pixels where BW is false to zero.

    maskedRGBImage[~np.dstack((BW, BW, BW))] = 0

    nRed = np.sum(BW)
    return nRed, BW

def findFluorescenceColor(RGB, mask):
    """
    Finds the fluorescence of a cell
    Input: RGB image location
    Output: Color
    """
    # RGB = imread(RGBLocation)
    mask = mask.astype('bool')
    RGB[~np.dstack((mask,mask,mask))] = 0
    nGreen, BW = cellMorphHelper.segmentGreen(RGB)
    nRed, BW = cellMorphHelper.segmentRed(RGB)
    if nGreen>=(nRed+100):
        return "green"
    elif nRed>=(nGreen+100):
        return "red"
    else:
        return "NaN"

def interpolatePerimeter(perim: np.array, nPts: int=150):
    """
    Interpolates a 2D curve to a given number of points. 
    Adapted from: https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
    Inputs:
    perim: 2D numpy array of dimension nptsx2
    nPts: Number of interpolated points
    Outputs:
    perimInt: Interpolated perimeter
    """
    distance = np.cumsum( np.sqrt(np.sum( np.diff(perim, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    alpha = np.linspace(0, 1, nPts)

    interpolator =  interp1d(distance, perim, kind='cubic', axis=0)
    perimInt = interpolator(alpha)
    
    return perimInt

def procrustes(X, Y, scaling=False, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def extractFeatures(image, mask, perim):
    """
    A wrapper function for pyfeats (https://github.com/giakou4/pyfeats) to extract parameters
    Inputs:
    f: A grayscale image scaled between 0 and 255
    mask: A mask of ints where the cell is located
    perim: The perimeter of the cell

    Outputs:
    allLabels: List of descriptors for each feature
    allFeatures: List of features for the given image
    """

    features = {}
    features['A_FOS']       = pyfeats.fos(image, mask)
    features['A_GLCM']      = pyfeats.glcm_features(image, ignore_zeros=True)
    features['A_GLDS']      = pyfeats.glds_features(image, mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
    features['A_NGTDM']     = pyfeats.ngtdm_features(image, mask, d=1)
    features['A_SFM']       = pyfeats.sfm_features(image, mask, Lr=4, Lc=4)
    features['A_LTE']       = pyfeats.lte_measures(image, mask, l=7)
    features['A_FDTA']      = pyfeats.fdta(image, mask, s=3)
    features['A_GLRLM']     = pyfeats.glrlm_features(image, mask, Ng=256)
    features['A_FPS']       = pyfeats.fps(image, mask)
    features['A_Shape_par'] = pyfeats.shape_parameters(image, mask, perim, pixels_per_mm2=1)
    features['A_HOS']       = pyfeats.hos_features(image, th=[135,140])
    features['A_LBP']       = pyfeats.lbp_features(image, image, P=[8,16,24], R=[1,2,3])
    features['A_GLSZM']     = pyfeats.glszm_features(image, mask)

    #% B. Morphological features
    # features['B_Morphological_Grayscale_pdf'], features['B_Morphological_Grayscale_cdf'] = pyfeats.grayscale_morphology_features(image, N=30)
    # features['B_Morphological_Binary_L_pdf'], features['B_Morphological_Binary_M_pdf'], features['B_Morphological_Binary_H_pdf'], features['B_Morphological_Binary_L_cdf'], \
    # features['B_Morphological_Binary_M_cdf'], features['B_Morphological_Binary_H_cdf'] = pyfeats.multilevel_binary_morphology_features(image, mask, N=30, thresholds=[25,50])
    #% C. Histogram Based features
    # features['C_Histogram'] =               pyfeats.histogram(image, mask, bins=32)
    # features['C_MultiregionHistogram'] =    pyfeats.multiregion_histogram(image, mask, bins=32, num_eros=3, square_size=3)
    # features['C_Correlogram'] =             pyfeats.correlogram(image, mask, bins_digitize=32, bins_hist=32, flatten=True)
    #% D. Multi-Scale features
    features['D_DWT'] =     pyfeats.dwt_features(image, mask, wavelet='bior3.3', levels=3)
    features['D_SWT'] =     pyfeats.swt_features(image, mask, wavelet='bior3.3', levels=3)
    # features['D_WP'] =      pyfeats.wp_features(image, mask, wavelet='coif1', maxlevel=3)
    features['D_GT'] =      pyfeats.gt_features(image, mask)
    features['D_AMFM'] =    pyfeats.amfm_features(image)

    #% E. Other
    # features['E_HOG'] =             pyfeats.hog_features(image, ppc=8, cpb=3)
    features['E_HuMoments'] =       pyfeats.hu_moments(image)
    # features['E_TAS'] =             pyfeats.tas_features(image)
    features['E_ZernikesMoments'] = pyfeats.zernikes_moments(image, radius=9)
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
    return allFeatures, allLabels

def alignPerimeters(cells: list):
    """
    Aligns a list of cells of class cellPerims
    Inputs:
    cells: A list of instances of cellPerims
    Ouputs:
    List with the instance variable perimAligned as an interpolated perimeter aligned
    to the first instance in list.
    """
    # Create reference perimeter from first 100 cells
    referencePerimX = []
    referencePerimY = []
    for cell in cells[0:1]:
        # Center perimeter
        originPerim = cell.perimInt.copy() - np.mean(cell.perimInt.copy(), axis=0)
        referencePerimX.append(originPerim[:,0])
        referencePerimY.append(originPerim[:,1])
    # Average perimeters
    referencePerim = np.array([ np.mean(np.array(referencePerimX), axis=0), \
                                np.mean(np.array(referencePerimY), axis=0)]).T

    # Align all cells to the reference perimeter
    c = 1
    for cell in cells:
        currentPerim = cell.perimInt
        
        # Perform procrustes to align orientation (not scaled by size)
        refPerim2, currentPerim2, disparity = procrustes(referencePerim, currentPerim, scaling=False)

        # Put cell centered at origin
        cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)
    return cells

def filterCells(cells, confluencyDate=False, edge=False, color=False):
    """
    Filters cells on commonly-occuring issues. 
    Inputs:
    cells: List of cells of class cellPerims
    confluencyDate: Datetime object, will filter cells before date
    edge: Checks if cell is split across multiple images
    color: Makes sure that fluorescence is correct
    """
    nCells = len(cells)
    if confluencyDate  != False:
        cells = [cell for cell in cells if cell.date < confluencyDate]
    if edge != False:
        cells = [cell for cell in cells if clearEdgeCells(cell) == 1]
    if color != False:
        cells = [cell for cell in cells if cell.color.lower() == color.lower()]
    nCellsNew = len(cells)
    print(f'Filtered out {nCells-nCellsNew} cells')
    return cells
# Testing
def validateExperimentData(experiment, splitNum=16):
    """
    Ensures that an experiment is properly set up without errant files. 
    Checks that:
    Experiment is split with a phase contrast and composite directory
    All files are .png format
    Each phase contrast has a composite twin
    """
    experiment+=f'Split{splitNum}'
    # Check that experiment folder is in proper format
    assert os.path.isdir(os.path.join('../data', experiment)), 'Split experiment before verifying'
    assert os.path.isdir(os.path.join('../data', experiment, 'phaseContrast')), 'No phase contrast directory'
    assert os.path.isdir(os.path.join('../data', experiment,'composite')), 'No composite directory'

    # Check that files are of correct extension
    phaseContrastFiles = os.listdir(os.path.join('../data', experiment, 'phaseContrast'))
    compositeFiles =     os.listdir(os.path.join('../data', experiment, 'composite'))
    maskDir = os.path.join('../data', experiment, 'mask')
    if os.path.isdir(maskDir):
        maskFiles =          os.listdir(maskDir)
    else:
        maskFiles = []

    phaseImageBases = []
    for phaseContrastFile in phaseContrastFiles:
        phaseImageBases.append(getImageBase(phaseContrastFile))
        assert phaseContrastFile.split('.')[-1] == 'png', f'Non .png file found:\n {phaseContrastFile}'

    for compositeFile in compositeFiles:
        imageBase = getImageBase(phaseContrastFile)
        assert compositeFile.split('.')[-1] == 'png', f'Non .png file found:\n {compositeFile}'
        assert imageBase in phaseImageBases, f'Composite file not in phase contrast files {compositeFile}'

    if len(maskFiles)>0:
        for maskFile in maskFiles:
            imageBase = getImageBase(maskFile)
            assert imageBase in phaseImageBases, f'Mask file not in phase contrast files {compositeFile}'
        
    print('Experimental data is in proper format!')
# Detectron2 Processes

def viewPredictorResult(predictor, imPath: str):
    """
    Plots an image of cells with masks overlaid.
    Inputs:
    predictor: A predictor trained with detectron2
    imPath: The path of the image to load
    Outputs:
    None
    """
    im = imread(imPath)
    imBase = getImageBase(imPath.split('/')[-1])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                #    metadata=cell_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure()
    print("plotting")
    plt.imshow(out.get_image()[:,:,::-1])
    plt.title(imBase)
    plt.show()

def getSegmentModel(modelPath: str, numClasses = 1):
    """
    Gets a segmentation model that can be used to output masks
    Inputs:
    modelPath: Folder with model. Final model must be named model_final.pth
    Outputs:
    Mask-RCNN model
    """
    cfg = get_cfg()
    if not torch.cuda.is_available():
        print('CUDA not available, resorting to CPU')
        cfg.MODEL.DEVICE='cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cellMorph_Train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = numClasses
    cfg.OUTPUT_DIR = modelPath
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    return predictor

# %%