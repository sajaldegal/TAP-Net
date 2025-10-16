import torch
import torch.nn as nn
import pandas as pd

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class TeacherNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3):
        super(TeacherNet, self).__init__()
        filters = [64, 128, 256, 512]
        self.encoder1 = ConvBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = ConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(filters[3], filters[3])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(filters[3], 3)
        )
        self.up1 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(filters[1] + filters[2], filters[1])
        self.up3 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(filters[0] + filters[1], filters[0])
        self.up4 = nn.ConvTranspose2d(filters[0], filters[0] // 2, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(filters[0] // 2, filters[0])
        self.segmentation_head = nn.Conv2d(filters[0], 1, kernel_size=1)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(filters[3], 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        plane_logits = self.classifier(b)
        d1 = self.up1(b)
        d1 = self.decoder1(torch.cat([d1, e4], dim=1))
        d2 = self.up2(d1)
        d2 = self.decoder2(torch.cat([d2, e3], dim=1))
        d3 = self.up3(d2)
        d3 = self.decoder3(torch.cat([d3, e2], dim=1))
        d4 = self.up4(d3)
        d4 = self.decoder4(d4)
        segmentation = self.segmentation_head(d4)
        value = self.regressor(b)
        return plane_logits, segmentation, value

def denormalize(value_norm, biom_type, min_max_path="biometric_min_max.csv"):
    min_max = pd.read_csv(min_max_path, index_col=0)
    type_map = {"head": ["HC", "BPD"], "abdomen": ["AC"], "femur": ["FL"]}
    types = type_map.get(biom_type.lower(), [])
    return {
        t: value_norm * (min_max.loc[t]["max"] - min_max.loc[t]["min"]) + min_max.loc[t]["min"]
        for t in types
    }