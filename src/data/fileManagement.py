"""
Tools for managing f
"""
import os
import numpy as np
import shutil
import pandas as pd
import datetime
import cv2
from tqdm import tqdm
import pickle
from pathlib import Path
import ast

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from src.data.imageProcessing import imSplit

def collateModelParameters(generate = False):
    """
    Collects model results
    Inputs:
        - generate: bool for generating, if false loads and prints results'
    Outputs:
        - defDetails: Dataframe of all experiments and details
    """
    outPath = Path(__file__).resolve().parent / '../../' / 'results' / 'classificationTraining'
    if generate == True:
        outFiles = [outPath / outFile for outFile in list(outPath.iterdir())]
        allModelDetails = {}
        for outFile in outFiles:
            modelDetails = getModelDetails(outFile)
            for detail in modelDetails.keys():
                if detail not in allModelDetails.keys():
                    allModelDetails[detail] = []

        for outFile in outFiles:
            modelDetails = getModelDetails(outFile)
            for detail in allModelDetails.keys():
                if detail not in modelDetails.keys():
                    allModelDetails[detail].append('')
                else:
                    allModelDetails[detail].append(modelDetails[detail])
                
        dfDetails = pd.DataFrame(allModelDetails)
        dfDetails.to_csv(outPath / '..' / 'allModelDetails.csv')
    else:
        dfDetailsPath = outPath / '..' / 'allModelDetails.csv'
        if not dfDetailsPath.exists():
            print('Generating results...')
            collateModelParameters(generate=True)
        dfDetails = pd.read_csv(dfDetailsPath, index_col=0)
    return dfDetails
def getModelDetails(outPath) -> dict:
    """
    Splits .out file from slurm and returns dictionary with details for recreating the model

    Inputs:
        - outPath: Location of .out file
    Outputs:
        - modelDetails: Dictionary of model parameters
    """
    print(outPath)
    with open(outPath) as outFile:
        x = outFile.read()
    detailsSplit = x.split('~ Model Details ~')[1]
    modelDetailsOut = detailsSplit.split('-'*10)[0]
    modelDetailsOut = modelDetailsOut.split('\n')[1:-1]
    modelDetails = {}
    for detail in modelDetailsOut:
        detail, val = detail.split(' - ')
        val = val.strip()
        if val.replace('.','').isnumeric():
            val = int(float(val))
        modelDetails[detail] = val
    if 'imgPaths' in modelDetails.keys():
        modelDetails['imgPaths'] = ast.literal_eval(modelDetails['imgPaths'])
    return modelDetails

def makeNewExperimentDirectory(experimentName):
    """
    # TODO: Completely rework for new structure
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

def splitExpIms(experiment, nIms=16):
    """
    # TODO: Completely rework for new structure
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

def splitName2Whole(imgName: str):
    """
    Strips split number from image name

    Inputs:
        - imgName: Name of image in format:
            imagingType_Well_IncucyteNum_Date_Time_ImgNum.extension
    Outputs:
        - imgNameWhole: Name of image in format:
            imagingType_Well_IncucyteNum_Date_Timeß.extension

    """
    extSplit = imgName.split('.')
    ext = extSplit[-1]
    imgName = extSplit[0]

    imgNameWhole = '_'.join(imgName.split('_')[0:-1])+'.'+ext
    return imgNameWhole

def datasetDict2Coco(datasetDicts, outputLocation, datasetName):
    def data_dict(datasetDicts):
        return datasetDicts
    inputs = [datasetDicts]
    DatasetCatalog.register(datasetName, lambda x=inputs: data_dict(inputs[0]))
    MetadataCatalog.get(datasetName).set(thing_classes=[0, 1])
    convert_to_coco_json(datasetName, output_file=outputLocation, allow_cached=False)

def loadDatasetDictJSON(datasetLocation):
    register_coco_instances('data', {}, "../data/sartorius/segmentations/train.json", "../data/sartorius/images/livecell_train_val_images")

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    DEPRECATED

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
