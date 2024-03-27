import numpy as np
import itertools
from scipy.interpolate import interp1d

from skimage.color import rgb2hsv
from skimage import morphology, measure
from skimage.segmentation import clear_border
from skimage.draw import polygon2mask, polygon_perimeter
import cv2
# import pyfeats

# %% General tools
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
    nGreen, BW = segmentGreenHigh(RGB)
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
        return RGB, 0
    isAbberation = np.isin(labels, numsHigh)
    # Use convex hull to fully enclose cells
    convexAbberation = morphology.convex_hull_image(isAbberation)
    convexAbberation = dilN(convexAbberation, 50)

    RGBNew = RGB.copy()
    RGBNew[convexAbberation, 1] = np.median(RGBNew[:,:,1])
    RGBNew[convexAbberation, 2] = np.median(RGBNew[:,:,2])
    
    return RGBNew, 1

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

def clearEdgeCells(cell):
    """
    Checks if cells are on border by dilating them and then clearing the border. 
    NOTE: This could be problematic since some cells are just close enough, but could be solved by stitching each image together, then checking the border.
    """
    mask = cell.mask
    maskDilate = morphology.binary_dilation(mask)
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

def segmentGreenHigh(RGB):
    """
    Finds very bright green pixels from Incucyte data
    Input: RGB image
    Output: # of green pixels and mask of green pixels
    """
    # def segment
    I = RGB.copy()

    # Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.000;
    channel1Max = 255.000;

    # Define thresholds for channel 2 based on histogram settings
    channel2Min = 198.000;
    channel2Max = 255.000;

    # Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.000;
    channel3Max = 84.000;

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
    RGB = RGB.copy()
    mask = mask.astype('bool')
    RGB[~np.dstack((mask,mask,mask))] = 0
    nGreen, BW = segmentGreen(RGB)
    nRed, BW = segmentRed(RGB)
    if nGreen>=(nRed+100):
        return "green"
    elif nRed>=(nGreen+100):
        return "red"
    else:
        return "NaN"

def findBrightGreen(RGB, mask, thresh = 10):
    """
    Finds the fluorescence of a cell
    Input: RGB image location
    Output: Color
    """
    RGB = RGB.copy()
    mask = mask.astype('bool')
    RGB[~np.dstack((mask,mask,mask))] = 0
    nGreen, BW = segmentGreenHigh(RGB)

    if thresh == 0:
        return nGreen
    elif nGreen>=thresh:
        return "green"
    else:
        return "NaN"
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

def preprocess(input_image, magnification_downsample_factor=1.0):
    """
    preprocesses an image taken by the Incucyte so that it will be normalized
    Taken from LIVECell paper: https://github.com/sartorius-research/LIVECell
    Inputs:
    input_image: Image to be analyzed
    magnification_downsample_factor: For other magnifications
    Outputs:
    output_image: Processed image
    """ 
    #internal variables
    #   median_radius_raw = used in the background illumination pattern estimation. 
    #       this radius should be larger than the radius of a single cell
    #   target_median = 128 -- LIVECell phase contrast images all center around a 128 intensity
    median_radius_raw = 75
    target_median = 128.0
    
    #large median filter kernel size is dependent on resize factor, and must also be odd
    median_radius = round(median_radius_raw*magnification_downsample_factor)
    if median_radius%2==0:
        median_radius=median_radius+1

    #scale so mean median image intensity is 128
    input_median = np.median(input_image)
    intensity_scale = target_median/input_median
    output_image = input_image.astype('float')*intensity_scale

    #define dimensions of downsampled image image
    dims = input_image.shape
    y = int(dims[0]*magnification_downsample_factor)
    x = int(dims[1]*magnification_downsample_factor)

    #apply resizing image to account for different magnifications
    output_image = cv2.resize(output_image, (x,y), interpolation = cv2.INTER_AREA)
    
    #clip here to regular 0-255 range to avoid any odd median filter results
    output_image[output_image > 255] = 255
    output_image[output_image < 0] = 0

    #estimate background illumination pattern using the large median filter
    background = cv2.medianBlur(output_image.astype('uint8'), median_radius)
    output_image = output_image.astype('float')/background.astype('float')*target_median

    #clipping for zernike phase halo artifacts
    output_image[output_image > 180] = 180
    output_image[output_image < 70] = 70
    output_image = output_image.astype('uint8')

    return output_image
# %% Expanding image segmentation to larger image
def split2WholeCoords(nIms, wholeImgSize):
    """
    Returns coordinates to connect split images to whole images

    Inputs:
        - nIms: This is the number of images an original image was split into
        - wholeImgSize: 1x2 list of type [nRows, nCols]

    Outputs:
        - coordinates: A dictionary where keys are the split number and 

    Example:
    coordinates = split2WholeCoords(nIms=16, wholeImg=img)
    # polyx and polyy are the initial segmentation coordinates
    polyxWhole = polyx + coordinates[int(splitNum)][0]
    polyyWhole = polyy + coordinates[int(splitNum)][1]
    """ 

    div = int(np.sqrt(nIms))
    nRow = wholeImgSize[0]
    nCol = wholeImgSize[1]

    M = nRow//div
    N = nCol//div
    tiles = []
    imNum = 1
    coordinates = {}
    for x in range(0,wholeImgSize[1],N): # Column
        for y in range(0,wholeImgSize[0],M): # Row
            coordinates[imNum] = [x, y]
            imNum += 1

    return coordinates

def expandImageSegmentation(poly, bb, splitNum, coords, padNum=200):
    """
    Takes segmentation information from split image and outputs it for a whole (non-split) image
    
    Inputs:
    - poly: Polygn of segmentation from datasetDict
    - bb: Bounding box of segmentation from datasetDict
    - padNum: Amount of padding around image
    - coords: Coordinates to relate split image to whole image

    Outputs:
    - polyxWhole, polyyWhole: polygon coordinates for whole image
    - bbWhole: bounding box for whole image
    """
    polyx = poly[:,0]
    polyy = poly[:,1]

    cIncrease = coords[int(splitNum)]
    bbWhole = bb.copy()
    bbWhole[1] += cIncrease[1] + padNum
    bbWhole[3] += cIncrease[1] + padNum
    bbWhole[0] += cIncrease[0] + padNum
    bbWhole[2] += cIncrease[0] + padNum

    polyxWhole = polyx + cIncrease[0] + padNum
    polyyWhole = polyy + cIncrease[1] + padNum
    
    return [polyxWhole, polyyWhole, bbWhole]

def bbIncrease(poly, bb, imgName, imgWhole, nIms, nIncrease=50, padNum=200, augmentation = None):
    """
    Takes in a segmentation from a split image and outputs the segmentation from the whole image. 
    Inputs: 
    - poly: Polygon in datasetDict format
    - bb: Bounding box in datasetDict format
    - imageName: Name of the image where the segmentation was found
    - imgWhole: The whole image from which the final crop will come from
    - nIncrease: The amount to increase the bounding box
    - padNum: The padding on the whole image, necessary to segment properly

    Outputs:
    - imgBBWholeExpand: The image cropped from the whole image increased by nIncrease
    """
    splitNum = int(imgName.split('_')[-1].split('.')[0])
    coords = split2WholeCoords(nIms, wholeImgSize = imgWhole.shape)
    imgWhole = np.pad(imgWhole, (padNum,padNum))
    polyxWhole, polyyWhole, bbWhole = expandImageSegmentation(poly, bb, splitNum, coords, padNum)
    bbWhole = [int(corner) for corner in bbWhole]
    colMin, rowMin, colMax, rowMax = bbWhole
    rowMin -= nIncrease
    rowMax += nIncrease
    colMin -= nIncrease
    colMax += nIncrease

    if augmentation == 'blackoutCell':
        maskBlackout  = polygon2mask(imgWhole.shape, np.array([polyyWhole, polyxWhole], dtype="object").T)

        imgWhole[maskBlackout] = 255

    if augmentation == 'outline':
            rr, cc = polygon_perimeter(polyyWhole, polyxWhole)
            imgWhole = np.zeros(imgWhole.shape)
            imgWhole[rr, cc] = 1

    if augmentation == 'stamp':
        maskBlackout  = polygon2mask(imgWhole.shape, np.array([polyyWhole, polyxWhole], dtype="object").T)

        imgWhole[~maskBlackout] = 0

    if augmentation == 'shape':
        maskBlackout  = polygon2mask(imgWhole.shape, np.array([polyyWhole, polyxWhole], dtype="object").T)
        imgWhole = maskBlackout


    bbIncrease = [colMin, rowMin, colMax, rowMax]
    imgBBWholeExpand = imgWhole[bbIncrease[1]:bbIncrease[3], bbIncrease[0]:bbIncrease[2]]
    return imgBBWholeExpand

def bbIncreaseBlackout(poly, bb, imgName, imgWhole, nIms, label, nIncrease=50, padNum=200):
    """
    Takes in a segmentation from a split image and outputs the segmentation from the whole image.
    This differs from bbIncrease because it will black out the cell.  
    Inputs: 
    - poly: Polygon in datasetDict format
    - bb: Bounding box in datasetDict format
    - imageName: Name of the image where the segmentation was found
    - imgWhole: The whole image from which the final crop will come from
    - nIncrease: The amount to increase the bounding box
    - padNum: The padding on the whole image, necessary to segment properly

    Outputs:
    - imgBBWholeExpand: The image cropped from the whole image increased by nIncrease where the cell is all black (0s)
    """
  

    splitNum = int(imgName.split('_')[-1].split('.')[0])
    coords = split2WholeCoords(nIms, wholeImgSize = imgWhole.shape)
    imgWhole = np.pad(imgWhole, (padNum,padNum))
    polyxWhole, polyyWhole, bbWhole = expandImageSegmentation(poly, bb, splitNum, coords, padNum)
    bbWhole = [int(corner) for corner in bbWhole]
    colMin, rowMin, colMax, rowMax = bbWhole
    rowMin -= nIncrease
    rowMax += nIncrease
    colMin -= nIncrease
    colMax += nIncrease
    
    maskBlackout  = polygon2mask(imgWhole.shape, np.array([polyyWhole, polyxWhole], dtype="object").T)

    imgWhole[maskBlackout] = 0

    
    bbIncrease = [colMin, rowMin, colMax, rowMax]
    imgBBWholeExpand = imgWhole[bbIncrease[1]:bbIncrease[3], bbIncrease[0]:bbIncrease[2]]
    

    return imgBBWholeExpand

# %% Perimeter and "classic" cell morphology
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

