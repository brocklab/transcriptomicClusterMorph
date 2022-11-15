# %%
import io
import numpy as np
import os

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Basically do the same thing as before
        # Input, Output, Convolutional size
        self.conv1 = nn.Conv2d(1, 32, 5) #inputs 1, outputs 32 using a 5x5
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        x = torch.randn(150,150).view(-1,1,150,150)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512) # Flattening
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 classes out
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] 
        return x
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear) # Recall that .view == reshape
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = Net()

print(model)

dumm_input = torch.randn(1,150,150)
torch.onnx.export(model, dumm_input, '../../output/smallConvnet.onnx')
# %%