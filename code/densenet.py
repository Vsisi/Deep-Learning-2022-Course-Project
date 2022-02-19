import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear
from dataset import trainData
from torch.utils.data import DataLoader
from torch.optim import Adam

class Bottleneck(nn.Module):
    def __init__(self, inc,  k=32) -> None:
        super().__init__()
        self.bn1 = BatchNorm2d(inc)
        self.relu = ReLU()
        self.conv1 = Conv2d(in_channels=inc, out_channels=4*k, kernel_size=1, stride=1, bias=False)

        self.bn2 = BatchNorm2d(4*k)
        self.relu = ReLU()
        self.conv2 = Conv2d(in_channels=4*k, out_channels=k, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, inputs):
        x = self.bn1(inputs)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return torch.cat([inputs, x], 1)

class DenseBlock(nn.Module):
    def __init__(self, inc, rep, k=32) -> None:
        super().__init__()
        self.rep = rep
        self.k0 = inc
        self.k = k

    def forward(self, inputs):
        x = inputs
        inc = self.k0
        for _ in range(self.rep):
            x = Bottleneck(inc, self.k)(x)
            inc = inc + self.k
        return x

class TransitionLayer(nn.Module):
    def __init__(self, inc, theta=0.5) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            BatchNorm2d(inc),
            ReLU(),
            Conv2d(in_channels=inc, out_channels=int(theta*inc), kernel_size=1, stride=1, bias=False)
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, initc, k=32, theta=0.5) -> None:
        super().__init__()
        self.init = nn.Sequential(
            Conv2d(in_channels=3, out_channels=initc, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(initc),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conf = [6, 12, 24, 16]
        
        self.db1 = DenseBlock(initc, self.conf[0], k)
        self.t1 = TransitionLayer(initc + self.conf[0]*k, theta)
        self.outc1 = int((initc + self.conf[0]*k)*theta)

        self.db2 = DenseBlock(self.outc1, self.conf[1], k)
        self.t2 = TransitionLayer(self.outc1 + self.conf[1]*k, theta)
        self.outc2 = int((self.outc1 + self.conf[1]*k)*theta)

        self.db3 = DenseBlock(self.outc2, self.conf[2], k)
        self.t3 = TransitionLayer(self.outc2 + self.conf[2]*k, theta)
        self.outc3 = int((self.outc2 + self.conf[2]*k)*theta)

        self.db4 = DenseBlock(self.outc3, self.conf[3], k)
        self.pool = AdaptiveAvgPool2d(1)
        self.fc = Linear(self.outc3 + self.conf[3]*k, 12)

    def forward(self, inputs):
        x = self.init(inputs)
        x = self.db1(x)
        x = self.t1(x)

        x = self.db2(x)
        x = self.t2(x)

        x = self.db3(x)
        x = self.t3(x)
        
        x = self.db4(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)

        return x

BATCH_SIZE = 16
dataloader = DataLoader(dataset=trainData, shuffle=True, batch_size=BATCH_SIZE)
device="cpu"

EPOCHS = 50
model = DenseNet(64); model.to(device); model.train()
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
