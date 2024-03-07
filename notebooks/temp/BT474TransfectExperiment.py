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
models = models[-3:]

for model in models:
    det = getModelDetails(f'../../results/classificationTraining/{model}.txt')