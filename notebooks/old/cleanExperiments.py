# %%
from src.data.fileManagement import collateModelParameters
from pathlib import Path
# %%
dfExperiment = collateModelParameters(generate=True)

# %%
resultsDir = Path('../results/classificationTraining')
modelDir   = Path('../models/classification')
for modelName, isEmpty in zip(dfExperiment['modelName'], dfExperiment['empty']):
    
    modelName = modelName.split('.pth')[0]





    if isEmpty:
        modelPath = modelDir / f'{modelName}.pth'
        
        resultsPath = resultsDir / f'{modelName}.txt'
        if not resultsPath.exists():
            resultsPath = resultsDir / f'{modelName}.out'
        
        assert resultsPath.exists()

        if modelPath.exists():
            print(modelName)
# %%
