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
models = ['classifySingleCellCrop-1709841378',
          'classifySingleCellCrop-1709844237',
          'classifySingleCellCrop-1709847182']
# models = models[-2:]
for model in models:
    det = getModelDetails(f'../../results/classificationTraining/{model}.txt')
    print(det['modelName'])
# %%
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
c = 2
labels = ['Transfect vs. LPD4', 'Transfect Green vs. Non', 'Transfect Non-Green vs. LPD4']
for model in models[-1:]:
    res = modelRes[model]
    plotLabel = f'AUC = {res["auc"]:0.2f}'
    plt.plot(res['fpr'], res['tpr'], linewidth = 3, label = labels[c])
    c +=1

plt.legend(loc = 'lower right', fontsize = 12.5)
# plt.savefig('../../figures/tempPres/transfectPred.png', dpi = 500)
# %%