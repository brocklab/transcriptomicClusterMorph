# %%
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, resnet152

from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

import torch
#%%
model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]
# %%
image = imread('../../data/misc/both.png')
input_tensor = torch.from_numpy(image.transpose([2,0,1])).float().unsqueeze(0)
# %%
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
# targets = [ClassifierOutputTarget(281)]
targets = None

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
# %%
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(image/255, grayscale_cam, use_rgb=True)
# %%
plt.imshow(visualization)
# %%
model2 = resnet152(pretrained=True)
target_layers2 = [model2.layer4[-1]]
# %%
