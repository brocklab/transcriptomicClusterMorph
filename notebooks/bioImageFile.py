# %%
import os
import pandas as pd
from pathlib import Path
# %%
dataDir = '../data/upload'
filePaths = []
annotations = []
annotationDict = {'TJ2201':         '231Subpop1 and 231Subpop2 coculture',
                  'TJ2454-2311kb32':'MDA-MB-231 untreated population',
                  'TJ2301-231C2':   'MDA-MB-231 treated population',
                  'TJ2453-436Co':   '436Subpop1 and 436Subpop2 coculture'}
i = 0

filePathsDel = []
for root, dirs, files in os.walk(dataDir):

    if len(files) < 1:
        continue
    
    for file in files:
        experiment = root.split('/')[3]
        if not file.endswith('.png'):
            continue
        filePath = os.path.join(root, file)
        filePaths.append(filePath)

        if 'composite' in filePath:
            filePath2 = filePath.replace('composite', 'phaseContrast')
            if not Path(filePath2).exists():
                filePathsDel.append(filePath)
            
        annotations.append(annotationDict[experiment])

# %%
assert len(filePathsDel) < 10
for filePath in filePathsDel:
    os.remove(filePath)
    
# %%
filePaths = [filePath.replace('../data/upload/', 'upload/') if filePath.startswith('..') else filePath for filePath in filePaths]
bioImage = pd.DataFrame({'Files': filePaths, 'Description': annotations})

bioImage.to_csv('../data/upload/fileList.tsv', sep = '\t', index=False)
# %%