from audioop import maxpp
from random import shuffle
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d, Dropout
from torch.optim import Adam
from dataset import trainData, evalData, testData
from torch.utils.data import DataLoader

EPOCHS = 10
BATCH_SIZE = 16

class VGG_VD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerCnt = 0
        # self.conf = [2, 2, 4, 4, 4] #19 weight layers
        self.conf = [1, 1, 2, 2, 2] #11 weight layers
        
        self.stage0 = self._getStage(3, 64, 0)
        self.stage1 = self._getStage(64, 128, 1)
        self.stage2 = self._getStage(128, 256, 2)
        self.stage3 = self._getStage(256, 512, 3)
        self.stage4 = self._getStage(512, 512, 4)

        self.classifier = nn.Sequential(
            Linear(7*7*512, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, 12) # Original output dimension should be 1000, but here we set it to 12 to deal with the 12 cat classification problem
        )

    def forward(self, inputs):
        x = self.stage0(inputs)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.classifier(x)
        return x


    def _getStage(self, inc, outc, stageId):
        stage = nn.Sequential()
        stage.add_module("layer%d conv" % self.layerCnt, Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding="same")); self.layerCnt = self.layerCnt + 1
        stage.add_module("layer%d bn" % self.layerCnt, BatchNorm2d(outc)); self.layerCnt = self.layerCnt + 1
        stage.add_module("layer%d relu" % self.layerCnt, ReLU()); self.layerCnt = self.layerCnt + 1

        for i in range(self.conf[stageId] - 1):
            stage.add_module("layer%d conv" % self.layerCnt, Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, stride=1, padding="same")); self.layerCnt = self.layerCnt + 1
            stage.add_module("layer%d bn" % self.layerCnt, BatchNorm2d(outc)); self.layerCnt = self.layerCnt + 1
            stage.add_module("layer%d relu" % self.layerCnt, ReLU()); self.layerCnt = self.layerCnt + 1

        stage.add_module("layer %d maxpool" % self.layerCnt, MaxPool2d(kernel_size=2, stride=2)); self.layerCnt = self.layerCnt + 1
        return stage

vgg = VGG_VD()
print(vgg)
lossFunc = nn.CrossEntropyLoss()
optimizer = Adam(lr=0.002, params=vgg.parameters())
dataloader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)
vgg.to(device)

for epoch in range(1, EPOCHS+1):
    runningLoss = 0.0
    for cnt, data in enumerate(dataloader):
        img, label = data
        img = img.to(device)
        label = label.to(device)

        vgg.train() 

        optimizer.zero_grad()
        pred = vgg(img)
        loss = lossFunc(pred, label)
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()
        
        if (cnt+1) % 10 == 0:
            print("Batch %d, Step %d, Loss: %lf" %(epoch, cnt+1, runningLoss / 10))
            runningLoss = 0
