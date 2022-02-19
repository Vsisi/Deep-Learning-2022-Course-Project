import torch
from torch.nn import Linear, Conv2d, BatchNorm2d, AdaptiveAvgPool2d, ReLU, Sigmoid, MaxPool2d
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import trainData
from torch.optim import Adam

class SEBottleneck(nn.Module):
    def __init__(self, inc, midc, outc, stride=1, downsample=False, reduceRatio=16):
        super().__init__()
        self.stride = stride
        self.inc = inc
        self.outc = outc
        self.needDownsample = downsample

        self.conv1 = Conv2d(in_channels=inc, out_channels=midc, kernel_size=1, stride=stride, bias=False) #fit different size
        self.bn1 = BatchNorm2d(midc)
        self.relu = ReLU()

        self.conv2 = Conv2d(in_channels=midc, out_channels=midc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(midc)
        
        self.conv3 = Conv2d(in_channels=midc, out_channels=outc, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm2d(outc)

        self.downsample = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=stride, bias=False), #fit different channels
            BatchNorm2d(outc)
        )

        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc1 = Linear(outc, outc//16)
        self.fc2 = Linear(outc//16, outc)
        self.sigmoid = Sigmoid()

        
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

        scale = self.avgpool(y)
        scale = torch.squeeze(scale, -1)
        scale = torch.squeeze(scale, -1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        scale = torch.unsqueeze(scale, -1)
        scale = torch.unsqueeze(scale, -1)

        y = y * scale

        if self.needDownsample:
            shortcut = self.downsample(shortcut)
        
        x = shortcut + y
        x = self.relu(x)
        return x


class SEResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conf = [3, 4, 6, 3]

        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)

        self.stage0 = self._getStage(64, 64, 256, self.conf[0])
        self.stage1 = self._getStage(256, 128, 512, self.conf[1], 2)
        self.stage2 = self._getStage(512, 256, 1024, self.conf[2], 2)
        self.stage3 = self._getStage(1024, 512, 2048, self.conf[3], 2)

        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(2048, 12)


    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)

        return x

    def _getStage(self, inc, midc, outc, rep, stride=1):
        stage = [SEBottleneck(inc, midc, outc, stride, True)]
        
        for _ in range(rep-1):
            stage.append(SEBottleneck(outc, midc, outc))

        return nn.Sequential(*stage)


device = "cuda:0"
seresnet = SEResNet50()
seresnet.to(device)
dataloader = DataLoader(dataset=trainData, batch_size=16, shuffle=True)
opt = Adam(lr=0.001, params=seresnet.parameters())
lossFc = nn.CrossEntropyLoss()

EPOCHS = 40

for epochs in range(EPOCHS):
    runningLoss = 0.0
    for i, data in enumerate(dataloader):
        inputs, label = data
        inputs = inputs.to(device)
        label = label.to(device)

        opt.zero_grad()
        pred = seresnet(inputs)
        loss = lossFc(pred, label)
        loss.backward()
        opt.step()

        runningLoss += loss.item()

        if (i+1) % 10 == 0:
            print("Epoch: %d, Step %d, Loss %lf" % (epochs+1, i+1, runningLoss/10))
            runningLoss = 0
