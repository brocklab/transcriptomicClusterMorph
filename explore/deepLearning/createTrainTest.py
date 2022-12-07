# %%
import random
import numpy as np
import os
import shutil
from tqdm import tqdm
# %%
testSize = 0.1
# %%
def shuffleLists(l, seed=1234):
    random.seed(seed)
    l = list(zip(*l))
    random.shuffle(l)
    return list(zip(*l))
# %%
fileDir = '../../data/TJ2201Split16SingleCell'

files = os.listdir(fileDir)

esamNeg = []
esamPos = []
for file in files:
    if 'esamNeg' in file:
        esamNeg.append(file)
    elif 'esamPos' in file:
        esamPos.append(file)

esamNeg, esamPos = shuffleLists([esamNeg, esamPos])

trainIdx = int(len(esamNeg)*(1-testSize))


esamNegTrain = esamNeg[0:trainIdx]
esamNegTest = esamNeg[trainIdx:]

esamPosTrain = esamPos[0:trainIdx]
esamPosTest = esamPos[trainIdx:]

os.makedirs(os.path.join(fileDir, 'train'), exist_ok=True)
os.makedirs(os.path.join(fileDir, 'test'), exist_ok=True)


# %%
for file in tqdm(esamNegTrain):
    shutil.move(os.path.join(fileDir, file), os.path.join(fileDir, 'train'))

for file in tqdm(esamNegTest):
    shutil.move(os.path.join(fileDir, file), os.path.join(fileDir, 'test'))


for file in tqdm(esamPosTrain):
    shutil.move(os.path.join(fileDir, file), os.path.join(fileDir, 'train'))

for file in tqdm(esamPosTest):
    shutil.move(os.path.join(fileDir, file), os.path.join(fileDir, 'test'))