# %%
import json
import matplotlib.pyplot as plt

from src.data.fileManagement import getModelDetails

# %%
json_file_loc = '../../results/classificationResults/bt474Experiments.json'
with open(json_file_loc, 'r') as json_file:
    modelRes = json.load(json_file)

# %%
models = list(modelRes.keys())
# models = ['classifySingleCellCrop-1709841378',
#           'classifySingleCellCrop-1709844237',
#           'classifySingleCellCrop-1709847182']
models = models[-2:]
for model in models:
    det = getModelDetails(f'../../results/classificationTraining/{model}.txt')
    print(det['modelName'])
# %%
for model in models:
    res = modelRes[model]
    plt.plot(res['fpr'], res['tpr'])
    print(res['auc'])
# %%
