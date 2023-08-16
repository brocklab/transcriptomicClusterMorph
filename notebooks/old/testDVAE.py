# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.data.fileManagement import convertDate
from src.models import modelTools
from pathlib import Path
import numpy as np
import time
import sys
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# %%
experiment  = 'TJ2201'
nIncrease   = 25
maxAmt      = 10000
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
optimizer = 'sgd'
notes = 'Full run without blackout'

modelID, idSource = modelTools.getModelID(sys.argv)
modelSaveName = Path(f'../models/classification/singleCellAutoencoder-{modelID}.pth')
resultsSaveName = Path(f'../results/classificationTraining/singleCellAutoencoder-{modelID}.txt')
modelInputs = {

'experiment'    : experiment, 
'nIncrease'     : nIncrease,
'maxAmt'        : maxAmt,
'batch_size'    : batch_size,
'num_epochs'    : num_epochs,
'modelType'     : modelType,
'modelName'     : modelSaveName.parts[-1],
'modelIDSource' : idSource,
'notes'         : notes,
'optimizer'     : optimizer,
'augmentation'  : 'stamp'
}

# %%

# %%
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
datasetDicts = [record for record in datasetDicts if len(record['annotations']) > 0]
# %%
# Check size
wellSize = {}
for seg in datasetDicts:
    well = seg['file_name'].split('_')[1]
    if well not in wellSize.keys():
        wellSize[well] = 0
    wellSize[well] += len(seg['annotations'])

sum(list(wellSize.values()))
# %%
data_transforms = {'train': transforms.Compose([]), 'test': transforms.Compose([])}

dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath,
                                               modelInputs
                                            )
# %%
img, labels = next(iter(dataloaders['test']))
plt.imshow(img[0][0].cpu().detach().numpy())
# %%
batch_size = 64
epochs = 50
seed = 1234
log_interval = 10
category = 60
# %%
ngf = 64
ndf = 64
nc = 1

def prior(K, alpha):
    """
    Prior for the model.
    :K: number of categories
    :alpha: Hyper param of Dir
    :return: mean and variance tensors
    """
    # ラプラス近似で正規分布に近似
    # Approximate to normal distribution using Laplace approximation
    a = torch.Tensor(1, K).float().fill_(alpha)
    mean = a.log().t() - a.log().mean(1)
    var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
    return mean.t(), var.t() # Parameters of prior distribution after approximation
class Dir_VAE2(nn.Module):
    def __init__(self):
        super(Dir_VAE2, self).__init__()

        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                self.step1  = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
                self.step2  = nn.LeakyReLU(0.2, inplace=True)

                self.step3  = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
                self.step4  = nn.BatchNorm2d(ndf * 2)
                self.step5  = nn.LeakyReLU(0.2, inplace=True)

                self.step6  = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)
                self.step7  = nn.BatchNorm2d(ndf * 4)
                self.step8  = nn.LeakyReLU(0.2, inplace=True)

                self.step9  = nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False)
                self.step10 = nn.LeakyReLU(0.2, inplace=True)

                self.batch124 = nn.BatchNorm2d(1024)
                self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

                self.step11 = nn.Conv2d(1024, 1024, 16, 2, 1)

                self.step12 = nn.Conv2d(1024, 1024, 2, 1, 0)
            def forward(self, x):
                x = self.step1(x)
                x = self.step2(x)

                x = self.step3(x)
                x = self.step4(x)
                x = self.step5(x)

                x = self.step6(x)
                x = self.step7(x)
                x = self.step8(x)

                x = self.step9(x)
                x = self.step10(x)

                x = self.step11(x)
                x = self.batch124(x)
                x = self.leakyrelu(x)

                x = self.step12(x)
                x = self.batch124(x)
                x = self.leakyrelu(x)

                return x
        class Decoder(nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()

                self.relu = nn.ReLU(True)
                self.batch124 = nn.BatchNorm2d(1024)

                self.stepa = nn.ConvTranspose2d(1024, 1024, 2, 1, 0)
                self.stepb = nn.ConvTranspose2d(1024, 1024, 16, 2, 1)

                self.step1 = nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False)
                self.step2 = nn.BatchNorm2d(ngf * 8)
                self.step3 = nn.ReLU(True)

                self.step4 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False)
                self.step5 = nn.BatchNorm2d(ngf * 4)
                self.step6 = nn.ReLU(True)

                self.step7 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
                self.step8 = nn.BatchNorm2d(ngf * 2)
                self.step9 = nn.ReLU(True)
                self.step10 = nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 0, bias=False)
                self.step11 = nn.Sigmoid()
            
            def forward(self, x):
                x = self.stepa(x)
                x = self.batch124(x)
                x = self.relu(x)

                x = self.stepb(x)
                x = self.batch124(x)
                x = self.relu(x)

                x = self.step1(x)
                x = self.step2(x)
                x = self.step3(x)
                
                x = self.step4(x)
                x = self.step5(x)
                x = self.step6(x)
                
                x = self.step7(x)
                x = self.step8(x)
                x = self.step9(x)
                x = self.step10(x)
                x = self.step11(x)
                return x
            
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, category)
        self.fc22 = nn.Linear(512, category)

        self.fc3 = nn.Linear(category, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        # Dir prior
        self.prior_mean, self.prior_var = map(nn.Parameter, prior(category, 0.3)) # 0.3 is a hyper param of Dirichlet distribution
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False


    def encode(self, x):
        conv = self.encoder(x);
        h1 = self.fc1(conv.view(-1, 1024))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, gauss_z):
        dir_z = F.softmax(gauss_z,dim=1) 
        # This variable (z) can be treated as a variable that follows a Dirichlet distribution (a variable that can be interpreted as a probability that the sum is 1)
        # Use the Softmax function to satisfy the simplex constraint
        # シンプレックス制約を満たすようにソフトマックス関数を使用
        h3 = self.relu(self.fc3(dir_z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(-1,1024,1,1)
        return self.decoder(deconv_input)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        mu, logvar = self.encode(x)
        gauss_z = self.reparameterize(mu, logvar) 
        # gause_z is a variable that follows a multivariate normal distribution
        # Inputting gause_z into softmax func yields a random variable that follows a Dirichlet distribution (Softmax func are used in decoder)
        dir_z = F.softmax(gauss_z,dim=1) # This variable follows a Dirichlet distribution
        return self.decode(gauss_z), mu, logvar, gauss_z, dir_z

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, K):
        beta = 1.0
        loss = nn.MSELoss()
        # BCE = nn.MSELoss(recon_x.view(-1, 22500), x.view(-1, 22500), reduction='sum')
        mseLoss = loss(recon_x.view(-1, 22500), x.view(-1, 22500))
        # ディリクレ事前分布と変分事後分布とのKLを計算
        # Calculating KL with Dirichlet prior and variational posterior distributions
        # Original paper:"Autoencodeing variational inference for topic model"-https://arxiv.org/pdf/1703.01488
        prior_mean = self.prior_mean.expand_as(mu)
        prior_var = self.prior_var.expand_as(logvar)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        var_division = logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = mu - prior_mean # μ_１ - μ_0
        diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - K)        
        return mseLoss + KLD
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Dir_VAE2().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# %%
class normalize01(object):
    """Normalizes image to be between 0 and 1"""

    def __init__(self):
        self.x = 2
    def __call__(self, img):
        imgNorm = (img - img.min())/(img.max() - img.min())
        return imgNorm
    
norm = normalize01()

# %%
for epoch in range(1, epochs + 1):
    # Train
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(dataloaders['train'])):
        data = data[:, 0:1, :, :]
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, gauss_z, dir_z = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar, category)
        loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            #print(f"gause_z:{gauss_z[0]}") # Variables following a normal distribution after Laplace approximation
            #print(f"dir_z:{dir_z[0]},SUM:{torch.sum(dir_z[0])}") # Variables that follow a Dirichlet distribution. This is obtained by entering gauss_z into the softmax function
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloaders['train'].dataset),
                100. * batch_idx / len(dataloaders['train']),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataloaders['train'].dataset)))
    torch.save(model.state_dict(), './dirVAE.pth')
# %%
model.load_state_dict(torch.load('./dirVAE.pth'))
# %%
c = 0
encodings, labels = [], []
for imgs, label in tqdm(dataloaders['train']):
    imgs = imgs[:, 0:1, :, :]
    imgs = imgs.to(device)

    recon_batch, mu, logvar, gauss_z, dir_z = model(imgs)
    encodings.append(gauss_z.cpu().detach().numpy())
    labels.append(label.cpu().detach().numpy())
    c +=1
    if c > 2:
        break
encodings = np.concatenate(encodings)
labels = np.concatenate(labels)
# %%
idx = 4
img0 = imgs[idx][0].detach().cpu().numpy()
label0 = label[idx].detach().cpu().numpy()
decoding0 = recon_batch[idx][0].detach().cpu().numpy()
plt.subplot(121)
plt.imshow(img0)
plt.subplot(122)
plt.imshow(decoding0)