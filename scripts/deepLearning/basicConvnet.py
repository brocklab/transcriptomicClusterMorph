# +
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
# %matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# -

assert torch.cuda.is_available()
device = torch.device("cuda:0")
print("running on GPU")
training_data = np.load('../../data/esamMonoSegmented/training_data.npy', allow_pickle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(150,150).view(-1,1,150,150)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
"basicConvnet.py" 118L, 3607C                                                                                            1,1           Top
# +
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
# %matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# -

assert torch.cuda.is_available()
device = torch.device("cuda:0")
print("running on GPU")
training_data = np.load('../../data/esamMonoSegmented/training_data.npy', allow_pickle=True)


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


# +
X = torch.Tensor([i[0] for i in training_data]).view(-1,150,150)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1  # Reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

# +
net = Net()
net = net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_function = nn.MSELoss()
MODEL_NAME = f"model-{int(time.time())}"

def fwd_pass(X, y, train=False):

    if train:
        net.zero_grad()
    outputs = net(X)
    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss

def test(size=32):
    X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = fwd_pass(X.view(-1, 1, 150, 150).to(device), y.to(device))
    return val_acc, val_loss

def train(net):
    BATCH_SIZE = 100
    EPOCHS = 30

    with open(f"../../data/esamMonoSegmented/logs/{MODEL_NAME}.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,150,150)
                batch_y = train_y[i:i+BATCH_SIZE]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                #print(f"Acc: {round(float(acc),2)}  Loss: {round(float(loss),4)}")
                #f.write(f"{MODEL_NAME},{round(time.time(),3)},train,{round(float(acc),2)},{round(float(loss),4)}\n")
                # just to show the above working, and then get out:
                if i % 10 == 0:
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")
            print(f'Loss: {loss:0.7f} \t Accuracy: {acc:0.4}')
train(net)