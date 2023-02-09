# %%
import matplotlib.pyplot as plt

# %%
def plotTrainingRes(filePath, plot=True, title=''):
    """
    Plots results from slurm.out file

    Inputs:
        - filePath: Path to results .out file
        - plot: Boolean flag for plotting or returning results instead
        - title: Title on plot
    """
    with open(filePath) as outFile:
        x = outFile.read()
    x = x.split('\n')
    trainLoss, trainAcc, testLoss, testAcc = [], [], [], []

    # Get results from file
    for line in x:
        if 'train Loss:' in line:
            trainLine = line.split('train Loss:')[1].split()
            trainLoss.append(float(trainLine[0]))
            trainAcc.append(float(trainLine[2]))
        if 'test Loss:' in line:
            testLine = line.split('test Loss:')[1].split()
            testLoss.append(float(testLine[0]))
            testAcc.append(float(testLine[2]))

    # Plot train/test loss/accuracy over epochs
    if plot:
        plt.figure(figsize=(11,7))
        plt.subplot(221)
        plt.plot(trainLoss)
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.subplot(222)
        plt.plot(trainAcc)
        plt.xlabel('Epoch')
        plt.ylabel('Train Accuracy')
        plt.subplot(223)
        plt.plot(testLoss)
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss')
        plt.subplot(224)
        plt.plot(testAcc)
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.suptitle(title)
        plt.show()
    return [trainLoss, trainAcc, testLoss, testAcc]

