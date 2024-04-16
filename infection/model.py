from monai.networks.nets import SwinUNETR
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet_2D(nn.Module):
    def __init__(self, in_ch=2, feature_ch=48, final_ch=4):
        super(SegNet_2D, self).__init__()
        self.swin = SwinUNETR(img_size=(512, 512),
                              in_channels=in_ch,
                              out_channels=final_ch,
                              spatial_dims=2,
                              feature_size=feature_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(final_ch, 1, kernel_size=3, stride=1, padding=1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        F = self.swin(x)
        out = self.conv(F)
        out = self.sig(out)
        return out, F


class SegNet_3D(nn.Module):
    def __init__(self, in_ch=2, feature_ch=36, final_ch=4):
        super(SegNet_3D, self).__init__()
        self.swin = SwinUNETR(img_size=(192, 192, 128),
                              in_channels=in_ch,
                              out_channels=final_ch,
                              spatial_dims=3,
                              feature_size=feature_ch)
        self.conv = nn.Sequential(
            nn.Conv3d(final_ch, final_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(final_ch, 1, kernel_size=3, stride=1, padding=1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        F = self.swin(x)
        out = self.conv(F)
        out = self.sig(out)
        return out, F
