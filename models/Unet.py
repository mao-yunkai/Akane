import numpy as np
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = down()
        self.down2 = down()
        self.down3 = down()
        self.down4 = down()
        self.up4 = up()
        self.up3 = up()
        self.up2 = up()
        self.up1 = up()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        return x

class down(nn.Module):
    def __init__(self):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x