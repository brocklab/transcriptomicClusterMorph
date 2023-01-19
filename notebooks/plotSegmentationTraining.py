# %%
import json
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

from scipy import interpolate
from scipy.signal import savgol_filter
# %%
AG2021Split16Data = [json.loads(line)
        for line in open('../models/AG2021Split16/metrics.json', 'r', encoding='utf-8')]
TJ2201Split16Data = [json.loads(line)
        for line in open('../models/TJ2201Split16/metrics.json', 'r', encoding='utf-8')]
# %%
def interpolateSignal(jsonFile, metric):
    """Reads json from detectron2 and returns an interpolated function for metric"""
    iterations = []
    metricResults = []
    for pt in jsonFile:
        if metric in pt.keys():
            iterations.append(pt['iteration'])
            metricResults.append(pt[metric])

    # Sort by iteration to get rid of any weirdness (iterations aren't always in the correct order?)
    xy = zip(iterations, metricResults)

    iterMetric = np.array(sorted(xy, key=lambda x: x[0]))

    # Aplly a little bit of smoothing
    filt = savgol_filter(iterMetric[:,1], 13, 3)

    fMetric = interpolate.interp1d(iterMetric[:,0], filt)
    return fMetric
metric = 'total_loss'
# %%
full, select = {}, {}

metrics = ['total_loss', 'mask_rcnn/accuracy', 'mask_rcnn/false_negative', 'mask_rcnn/false_positive']
for metric in metrics:
    full[metric] = interpolateSignal(TJ2201Split16Data, metric)
    select[metric] = interpolateSignal(AG2021Split16Data, metric)

# %%
matplotlib.rcParams.update({'font.size': 12})
iters = np.linspace(20, 10000-1, 1000)
c = 1
plt.figure(figsize=(16,11))
for metric in metrics:
    plt.subplot(2,2,c)
    plt.plot(iters, full[metric](iters), label='Full Segmentation')
    plt.plot(iters, select[metric](iters), label = 'Manual Selection')
    plt.grid()
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel(metric)
    c+=1
plt.savefig('../results/figs/maskRCNNSegMetrics.png', dpi=600)
# %%
