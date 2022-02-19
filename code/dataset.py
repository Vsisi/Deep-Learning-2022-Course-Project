import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
from PIL import Image

baseDir = r'C:\sundries\homework\DL\dataset'

class Cat12(Dataset):
    def __init__(self, transform, mode):
        super().__init__()

        self.imgPath = []
        self.imgLabel = []
        self.mode = mode
        self.transform = transform

        with open(os.path.join(baseDir, mode+".txt"), "r") as fp:
            lines = fp.readlines()

            for line in lines:
                path, label = line.split(" ")
                self.imgPath.append(path)
                self.imgLabel.append(int(label))


    def __getitem__(self, idx):
        if self.mode != "test":
            path = os.path.join(baseDir, self.imgPath[idx])
        else:
            path = os.path.join(baseDir, "cat_12_test", self.imgPath[idx])
        im = Image.open(path).convert("RGB")
        img = self.transform(im)
        return img, self.imgLabel[idx]

    def __len__(self):
        return len(self.imgPath)



transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]
    )

trainData = Cat12(transform, "train")
evalData = Cat12(transform, "eval")
testData = Cat12(transform, "test")