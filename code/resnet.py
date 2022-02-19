import torch
import torch.nn as nn
from dataset import trainData
from torch.utils.data import DataLoader
from torch.nn import Conv2d, ReLU, BatchNorm2d, Linear, AdaptiveAvgPool2d, MaxPool2d
from torch.optim import Adam

BATCH_SIZE = 16
dataloader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle = True)

class Bottleneck(nn.Module):
    def __init__(self, inc, midc, outc, stride=1, downsample=False):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.stride = stride
        self.needDownsample = downsample

        self.conv1 = Conv2d(in_channels=inc, out_channels=midc, kernel_size=1, stride=stride,  bias=False) #fit dim
        self.bn1 = BatchNorm2d(midc)
        self.relu = ReLU()

        self.conv2 = Conv2d(in_channels=midc, out_channels=midc, kernel_size=3, stride=1, padding="same", bias=False)
        self.bn2 = BatchNorm2d(midc)

        self.conv3 = Conv2d(in_channels=midc, out_channels=outc, kernel_size=1, stride=1, padding="same", bias=False)
        self.bn3 = BatchNorm2d(outc)

        self.downsample = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=stride, bias=False), #fit channel
            BatchNorm2d(outc)
        )

    
    def forward(self, inputs):
        shortcut = inputs
        y = self.conv1(inputs)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)
        
        if self.needDownsample:
            shortcut = self.downsample(shortcut)

        y += shortcut

        x = self.relu(y)

        return x

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.stageRep = [3, 4, 6, 3]
        # self.stageRep = [3, 4, 23, 3] #ResNet 101
        # self.stageRep = [3, 8, 36, 3] #ResNet 152


        self.conv1 = Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()
        self.maxPool1 = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._getStage(64, 64, 256, self.stageRep[0])
        self.conv3_x = self._getStage(256, 128, 512, self.stageRep[1], 2)
        self.conv4_x = self._getStage(512, 256, 1024, self.stageRep[2], 2)
        self.conv5_x = self._getStage(1024, 512, 2048, self.stageRep[3], 2)

        self.avgPool = AdaptiveAvgPool2d(1)
        self.fc = Linear(2048, 12)


    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxPool1(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        
        x = self.avgPool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)
        return x


    def _getStage(self, inc, midc, outc, repeat, stride=1):
        stage = [Bottleneck(inc, midc, outc, stride, True)]

        for _ in range(repeat-1):
            stage.append(Bottleneck(outc, midc, outc))
        
        return nn.Sequential(*stage)

device = "cuda:0"
EPOCHS = 10

resnet = ResNet50()
resnet.to(device)
print(resnet)

optimizer = Adam(lr=0.001, params=resnet.parameters())
lossFc = nn.CrossEntropyLoss()

resnet.train()

for epoch in range(EPOCHS):
    runningLoss = 0
    for i, data in enumerate(dataloader):
        inputs, label = data
        inputs = inputs.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = resnet(inputs)
        loss = lossFc(pred, label)
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()

        if (i+1) % 10 == 0:
            print("Epoch %d, Step %d, Loss %lf" % (epoch+1, i+1, runningLoss/10))
            runningLoss = 0
