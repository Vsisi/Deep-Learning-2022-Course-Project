import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Cat12
from torch.nn import Dropout, Conv2d, BatchNorm2d, Linear, MaxPool2d, AdaptiveAvgPool2d, ReLU, AvgPool2d
from torchvision import transforms
from torch.optim import RMSprop

#omit auxillary classifier here, since it has a minor effet on predict precision (only 0.2%)

BATCH_SIZE = 16
transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])
trainData = Cat12(transform, "train")
dataloader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle = True)

class InceptionA(nn.Module):
    def __init__(self, inc, outc1, outc21, outc22, outc31, outc32, outc4):
        super().__init__()
        self.branch1 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc1, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc1),
            ReLU()
        )
        self.branch2 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc21, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc21),
            ReLU(),
            Conv2d(in_channels=outc21, out_channels=outc22, kernel_size=1, stride=1, padding="same", bias=False),
            BatchNorm2d(outc22),
            ReLU()
        )
        self.branch3 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc31, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc31),
            ReLU(),
            Conv2d(in_channels=outc31, out_channels=outc32, kernel_size=3, stride=1, padding="same", bias=False),
            BatchNorm2d(outc32),
            ReLU(),
            Conv2d(in_channels=outc32, out_channels=outc32, kernel_size=3, stride=1, padding="same", bias=False),
            BatchNorm2d(outc32),
            ReLU()
        )
        self.branch4 = nn.Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d(in_channels=inc, out_channels=outc4, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc4),
            ReLU()
        )
    def forward(self, inputs):
        b1 = self.branch1(inputs)
        b2 = self.branch2(inputs)
        b3 = self.branch3(inputs)
        b4 = self.branch4(inputs)
        return torch.cat([b1, b2, b3, b4], dim=1)

class InceptionB(nn.Module):
    def __init__(self, inc, outc1, outc21, outc22):
        super().__init__()
        self.branch1 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc1, kernel_size=3, stride=2, bias=False),
            BatchNorm2d(outc1),
            ReLU()
        )
        self.branch2 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc21, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc21),
            ReLU(),
            Conv2d(in_channels=outc21, out_channels=outc22, kernel_size=3, stride=1, padding="same", bias=False),
            BatchNorm2d(outc22),
            ReLU(),
            Conv2d(in_channels=outc22, out_channels=outc22, kernel_size=3, stride=2, bias=False),
            BatchNorm2d(outc22),
            ReLU()
        )
        self.branch3 = MaxPool2d(kernel_size=3, stride=2)

    def forward(self, inputs):
        b1 = self.branch1(inputs)
        b2 = self.branch2(inputs)
        b3 = self.branch3(inputs)
        return torch.cat([b1, b2, b3], dim=1)

class InceptionC(nn.Module):
    def __init__(self, inc, outc1, outc21, outc22, outc31, outc32, outc4):
        super().__init__()
        self.branch1 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc1, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc1),
            ReLU()
        )
        self.branch2 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc21, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc21),
            ReLU(),
            Conv2d(in_channels=outc21, out_channels=outc21, kernel_size=[1,7], stride=1, padding=[0, 3], bias=False),
            BatchNorm2d(outc21),
            ReLU(),
            Conv2d(in_channels=outc21, out_channels=outc22, kernel_size=[7, 1], stride=1, padding=[3, 0], bias=False),
            BatchNorm2d(outc22),
            ReLU()
        )
        self.branch3 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc31, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc31),
            ReLU(),
            Conv2d(in_channels=outc31, out_channels=outc31, kernel_size=[7, 1], stride=1, padding=[3, 0], bias=False),
            BatchNorm2d(outc31),
            ReLU(),
            Conv2d(in_channels=outc31, out_channels=outc31, kernel_size=[1, 7], stride=1, padding=[0, 3], bias=False),
            BatchNorm2d(outc31),
            ReLU(),
            Conv2d(in_channels=outc31, out_channels=outc31, kernel_size=[7, 1], stride=1, padding=[3, 0], bias=False),
            BatchNorm2d(outc31),
            ReLU(),
            Conv2d(in_channels=outc31, out_channels=outc32, kernel_size=[1, 7], stride=1, padding=[0, 3], bias=False),
            BatchNorm2d(outc32),
            ReLU()
        )
        self.branch4 = nn.Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d(in_channels=inc, out_channels=outc4, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc32),
            ReLU()
        )
    def forward(self, inputs):
        b1 = self.branch1(inputs)
        b2 = self.branch2(inputs)
        b3 = self.branch3(inputs)
        b4 = self.branch4(inputs)
        return torch.cat([b1, b2, b3, b4], dim=1)

class InceptionD(nn.Module):
    def __init__(self, inc, outc11, outc12, outc2) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc11, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc11),
            ReLU(),
            Conv2d(in_channels=outc11, out_channels=outc12, kernel_size=3, stride=2, bias=False),
            BatchNorm2d(outc12),
            ReLU()
        )

        self.branch2 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc2, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc2),
            ReLU(),
            Conv2d(in_channels=outc2, out_channels=outc2, kernel_size=[1, 7], stride=1, padding=[0, 3], bias=False),
            BatchNorm2d(outc2),
            ReLU(),
            Conv2d(in_channels=outc2, out_channels=outc2, kernel_size=[7, 1], stride=1, padding=[3, 0], bias=False),
            BatchNorm2d(outc2),
            ReLU(),
            Conv2d(in_channels=outc2, out_channels=outc2, kernel_size=3, stride=2, bias=False),
            BatchNorm2d(outc2),
            ReLU()
        )

        self.branch3 = MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, inputs):
        b1 = self.branch1(inputs)
        b2 = self.branch2(inputs)
        b3 = self.branch3(inputs)
        return torch.cat([b1, b2, b3], dim=1)

class InceptionE(nn.Module):
    def __init__(self, inc, outc1, outc2, outc31, outc32, outc4) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc1, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc1),
            ReLU()
        )

        
        self.branch2 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc2, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc2),
            ReLU()
        )
        self.subBranch21 = nn.Sequential(
            Conv2d(in_channels=outc2, out_channels=outc2, kernel_size=[1,3], stride=1, padding=[0, 1], bias=False),
            BatchNorm2d(outc2),
            ReLU()
        )
        self.subBranch22 = nn.Sequential(
            Conv2d(in_channels=outc2, out_channels=outc2, kernel_size=[3,1], stride=1, padding=[1, 0], bias=False),
            BatchNorm2d(outc2),
            ReLU()
        )

        self.branch3 = nn.Sequential(
            Conv2d(in_channels=inc, out_channels=outc31, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc31),
            ReLU(),
            Conv2d(in_channels=outc31, out_channels=outc32, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc32),
            ReLU()
        )
        self.subBranch31 = nn.Sequential(
            Conv2d(in_channels=outc32, out_channels=outc32, kernel_size=[1,3], stride=1, padding=[0, 1], bias=False),
            BatchNorm2d(outc32),
            ReLU()
        )
        self.subBranch32 = nn.Sequential(
            Conv2d(in_channels=outc32, out_channels=outc32, kernel_size=[3, 1], stride=1, padding=[1, 0], bias=False),
            BatchNorm2d(outc32),
            ReLU()
        )

        self.branch4 = nn.Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d(in_channels=inc, out_channels=outc4, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(outc4),
            ReLU()
        )
    def forward(self, inputs):
        b1 = self.branch1(inputs)

        b2 = self.branch2(inputs)
        sb21 = self.subBranch21(b2)
        sb22 = self.subBranch22(b2)

        b3 = self.branch3(inputs)
        sb31 = self.subBranch31(b3)
        sb32 = self.subBranch32(b3)

        b4 = self.branch4(inputs)

        return torch.cat([b1, sb21, sb22, sb31, sb32, b4], dim=1)



class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre0 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, bias=False),
            BatchNorm2d(32),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=False),
            BatchNorm2d(32),
            ReLU(),
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2)
        )

        self.pre1 = nn.Sequential(
            Conv2d(in_channels=64, out_channels=80, kernel_size=3, stride=1, bias=False),
            BatchNorm2d(80),
            ReLU(),
            Conv2d(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(192),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2)
        )


        self.inceptionA = nn.Sequential(
            InceptionA(192, 64, 48, 64, 64, 96, 32),
            InceptionA(256, 64, 48, 64, 64, 96, 64),
            InceptionA(288, 64, 48, 64, 64, 96, 64)
        )

        self.inceptionB = nn.Sequential(
            InceptionB(288, 384, 64, 96),
            InceptionC(768, 192, 128, 192, 128, 192, 192),
            InceptionC(768, 192, 160, 192, 160, 192, 192),
            InceptionC(768, 192, 160, 192, 160, 192, 192),
            InceptionC(768, 192, 192, 192, 192, 192, 192)
        )

        self.inceptionC = nn.Sequential(
            InceptionD(768, 192, 320, 192),
            InceptionE(1280, 320, 384, 448, 384, 192),
            InceptionE(2048, 320, 384, 448, 384, 192),
        )

        self.pool = AdaptiveAvgPool2d(1)
        self.dropout = Dropout(0.4)
        self.fc = Linear(2048, 12)
        

    def forward(self, inputs):
        x = self.pre0(inputs)
        x = self.pre1(x)
        x = self.inceptionA(x)
        x = self.inceptionB(x)
        x = self.inceptionC(x)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

EPOCHS = 50
device = "cuda:0"
model = InceptionV3(); model.to(device)
lossFc = nn.CrossEntropyLoss()
opt = RMSprop(lr=0.045, weight_decay=0.9, eps=1.0, params=model.parameters())

for epochs in range(EPOCHS):
    runningloss = 0.0
    for i, data in enumerate(dataloader):
        inputs, label = data
        inputs = inputs.to(device)
        label = label.to(device)

        opt.zero_grad()
        pred = model(inputs)
        loss = lossFc(pred, label)
        loss.backward()
        opt.step()

        runningloss = runningloss + loss.item()

        if (i+1) % 10 == 0:
            print("Epoch %d, Step %d, loss %lf" % (epochs+1, i+1, runningloss/10))
            runningloss = 0.0