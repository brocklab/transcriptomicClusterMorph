# %%
import os
import matplotlib.pyplot as plt
# %%

def transferLearningRes(filePath, title, plot=True):
    with open(filePath) as file:
        x = file.read()
    x = x.split('\n')
    trainLoss, trainAcc, testLoss, testAcc = [], [], [], []
    for line in x:
        if 'train Loss:' in line:
            trainLine = line.split('train Loss:')[1].split()
            trainLoss.append(float(trainLine[0]))
            trainAcc.append(float(trainLine[2]))
        if 'test Loss:' in line:
            testLine = line.split('test Loss:')[1].split()
            testLoss.append(float(testLine[0]))
            testAcc.append(float(testLine[2]))

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
# %%
title='resnet18 Pretrained'
resRes18Pre = transferLearningRes('../results/TJ2201Split16SingleCell/resnet18Pretrained.out', title=title)
title = 'resent152 Pretrained'
resRes152Pre = transferLearningRes('../results/TJ2201Split16SingleCell/resnet152Pretrained.out', title=title)
# %%
title = 'resnet152 Not Pretrained'
resRes152NotPre = transferLearningRes('../results/TJ2201Split16SingleCell/resnet152ESAMNoPretrain.out', title=title)

# %%
experiments = [resRes18Pre, resRes152Pre, resRes152NotPre]
experimentLabels = ['resnet18 (Pretrained)', 'resnet152 (Pretrained)', 'resnet152 (Not Pretrained)']
idxLabels = ['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy']
idx = 3
for idx in range(4):
    plt.figure(figsize=(7,7))
    plt.rcParams.update({'font.size': 18})
    for experimentLabel, experiment in zip(experimentLabels, experiments):
        plt.plot(experiment[idx], linewidth=4, label=experimentLabel)

    plt.grid()
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel(idxLabels[idx])
    figName = ''.join(idxLabels[idx].split())
    plt.savefig(f'../figures/temp/esamSingleCell{figName}.png',dpi=600, bbox_inches='tight')
# %%
resnetSplit = transferLearningRes('../results/TJ2201Split16SingleCell/splitESAMresnet152Pretrain.out', title='resnet 152 (Pretrained) Split Images', plot=True)
