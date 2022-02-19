from random import shuffle
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, BatchNorm2d, AdaptiveAvgPool2d, Sigmoid
from dataset import trainData
from torch.utils.data import DataLoader
from torch.optim import Adam

class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * Sigmoid()(x)

class MBConv(nn.Module):
    def __init__(self, inc, outc, k, stride, padding, expRatio=1, redRatio=16) -> None:
        super().__init__()
        self.expRatio = expRatio
        self.stride = stride
        self.inc = inc
        self.outc = outc

        midc = expRatio * inc
        self.conv1 = Conv2d(in_channels=inc, out_channels=midc, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm2d(midc)
        self.swish1 = Swish()

        self.conv2 = Conv2d(in_channels=midc, out_channels=midc, groups=midc, kernel_size=k, stride=stride, padding=padding, bias=False)
        self.bn2 = BatchNorm2d(midc)
        self.swish2 = Swish()

        self.pool = AdaptiveAvgPool2d(1)
        self.fc1 = Linear(midc, midc//redRatio)
        self.swish3 = Swish()
        self.fc2 = Linear(midc//redRatio, midc)
        self.sigmoid = Sigmoid()
        
        self.conv3 = Conv2d(in_channels=midc, out_channels=outc, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm2d(outc)


    def forward(self, inputs):
        x = inputs
        if self.expRatio != 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.swish1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.swish2(x)

        #se
        scale = self.pool(x)
        scale = torch.squeeze(scale, -1)
        scale = torch.squeeze(scale, -1)
        scale = self.fc1(scale)
        scale = self.swish3(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        scale = torch.unsqueeze(scale, -1)
        scale = torch.unsqueeze(scale, -1)
        x = x * scale

        x = self.conv3(x)
        x = self.bn3(x)

        if self.stride == 1 and self.inc == self.outc:
            x = x + inputs
        return x

class EfficientNetB0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conf = [1, 2, 2, 3, 3, 4, 1]
        
        self.stage1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(32),
            Swish()
        )

        self.stage2 = self._getStage(32, 16, 3, 1, self.conf[0], False)
        self.stage3 = self._getStage(16, 24, 3, 6, self.conf[1], True)
        self.stage4 = self._getStage(24, 40, 5, 6, self.conf[2], True)
        self.stage5 = self._getStage(40, 80, 3, 6, self.conf[3], False)
        self.stage6 = self._getStage(80, 112, 5, 6, self.conf[4], True)
        self.stage7 = self._getStage(112, 192, 5, 6, self.conf[5], True)
        self.stage8 = self._getStage(192, 320, 3, 6, self.conf[6], False)

        self.stage9 = nn.Sequential(
            Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(1280),
            Swish()
        )
        self.pool = AdaptiveAvgPool2d(1)
        self.fc = Linear(1280, 12)

    def forward(self, inputs):
        x = self.stage1(inputs)
        # print(x.shape)
        
        x = self.stage2(x)
        # print(x.shape)
        x = self.stage3(x)
        # print(x.shape)
        x = self.stage4(x)
        # print(x.shape)
        x = self.stage5(x)
        # print(x.shape)
        x = self.stage6(x)
        # print(x.shape)
        x = self.stage7(x)
        # print(x.shape)
        x = self.stage8(x)
        # print(x.shape)

        x = self.stage9(x)
        # print(x.shape)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)
        return x

    def _getStage(self, inc, outc, kernel_size, expandRatio, rep, down):
        stage = []
        padding = (kernel_size-1)//2
        if down:
            if rep == 1:
                stage.append(MBConv(inc, outc, kernel_size, 2, padding, expandRatio))
            else:
                stage.append(MBConv(inc, outc, kernel_size, 1, padding, expandRatio))
                for _ in range(rep-2):
                    stage.append(MBConv(outc, outc, kernel_size, 1, padding, expandRatio))
                stage.append(MBConv(outc, outc, kernel_size, 2, padding, expandRatio))
        else:
            stage.append(MBConv(inc, outc, kernel_size, 1, padding, expandRatio))
            for _ in range(rep-1):
                stage.append(MBConv(outc, outc, kernel_size, 1, padding, expandRatio))

        return nn.Sequential(*stage)


EPOCHS = 50
BATCH_SIZE = 16
device = "cuda:0"

model = EfficientNetB0(); model.to(device)
dataloader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
optim = Adam(lr=0.001, params=model.parameters())
lossFc = nn.CrossEntropyLoss()

for epochs in range(EPOCHS):
    runningLoss = 0.0
    for i, data in enumerate(dataloader):
        inputs, label = data
        inputs = inputs.to(device)
        label = label.to(device)

        optim.zero_grad()
        pred = model(inputs)
        loss = lossFc(pred, label)
        loss.backward()
        optim.step()

        runningLoss = runningLoss + loss.item()

        if (i+1) % 10 == 0:
            print("Epoch %d, Step %d, loss %lf" % (epochs+1, i+1, runningLoss/10))
            runningLoss = 0.0