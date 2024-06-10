# %%
import os
import pandas as pd
# %%
dataDir = '../data/upload'
filePaths = []
annotations = []
annotationDict = {'TJ2201':         '231Subpop1 and 231Subpop2 coculture',
                  'TJ2454-2311kb32':'MDA-MB-231 untreated population',
                  'TJ2301-231C2':   'MDA-MB-231 treated population',
                  'TJ2453-436Co':   '436Subpop1 and 436Subpop2 coculture'}
i = 0
for root, dirs, files in os.walk(dataDir):

    if len(files) < 1:
        continue
    
    for file in files:
        experiment = root.split('/')[3]
        if not file.endswith('.png'):
            continue
        filePath = os.path.join(root, file)
        filePaths.append(filePath)

        annotations.append(annotationDict[experiment])


    
    