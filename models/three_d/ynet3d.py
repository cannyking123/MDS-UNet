import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
# from sync_batchnorm import SynchronizedBatchNorm3d
# from torchsummary import summary


class UNet3DDualDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
               单编码器双解码器的UNet3D实现
               - 一个编码器路径
               - 两个独立的解码器路径
               - 两个独立的输出
        """

        super(UNet3DDualDecoder, self).__init__()

        features = init_features
        # 编码器部分 - 共享
        self.encoder1 = UNet3DDualDecoder._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3DDualDecoder._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3DDualDecoder._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3DDualDecoder._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3DDualDecoder._block(features * 8, features * 16, name="bottleneck")

        # 第一个解码器路径
        self.upconv4_1 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4_1 = UNet3DDualDecoder._block((features * 8) * 2, features * 8, name="dec4_1")
        self.upconv3_1 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3_1 = UNet3DDualDecoder._block((features * 4) * 2, features * 4, name="dec3_1")
        self.upconv2_1 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2_1 = UNet3DDualDecoder._block((features * 2) * 2, features * 2, name="dec2_1")
        self.upconv1_1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1_1 = UNet3DDualDecoder._block(features * 2, features, name="dec1_1")
        self.conv_1 = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
        # 第二个解码器路径
        self.upconv4_2 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4_2 = UNet3DDualDecoder._block((features * 8) * 2, features * 8, name="dec4_2")
        self.upconv3_2 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3_2 = UNet3DDualDecoder._block((features * 4) * 2, features * 4, name="dec3_2")
        self.upconv2_2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2_2 = UNet3DDualDecoder._block((features * 2) * 2, features * 2, name="dec2_2")
        self.upconv1_2 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1_2 = UNet3DDualDecoder._block(features * 2, features, name="dec1_2")
        self.conv_2 = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器路径
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        # 第一个解码器路径
        dec4_1 = self.upconv4_1(bottleneck)
        dec4_1 = torch.cat((dec4_1, enc4), dim=1)
        dec4_1 = self.decoder4_1(dec4_1)

        dec3_1 = self.upconv3_1(dec4_1)
        dec3_1 = torch.cat((dec3_1, enc3), dim=1)
        dec3_1 = self.decoder3_1(dec3_1)

        dec2_1 = self.upconv2_1(dec3_1)
        dec2_1 = torch.cat((dec2_1, enc2), dim=1)
        dec2_1 = self.decoder2_1(dec2_1)

        dec1_1 = self.upconv1_1(dec2_1)
        dec1_1 = torch.cat((dec1_1, enc1), dim=1)
        dec1_1 = self.decoder1_1(dec1_1)

        output1 = self.conv_1(dec1_1)

        # 第二个解码器路径
        dec4_2 = self.upconv4_2(bottleneck)
        dec4_2 = torch.cat((dec4_2, enc4), dim=1)
        dec4_2 = self.decoder4_2(dec4_2)

        dec3_2 = self.upconv3_2(dec4_2)
        dec3_2 = torch.cat((dec3_2, enc3), dim=1)
        dec3_2 = self.decoder3_2(dec3_2)

        dec2_2 = self.upconv2_2(dec3_2)
        dec2_2 = torch.cat((dec2_2, enc2), dim=1)
        dec2_2 = self.decoder2_2(dec2_2)

        dec1_2 = self.upconv1_2(dec2_2)
        dec1_2 = torch.cat((dec1_2, enc1), dim=1)
        dec1_2 = self.decoder1_2(dec1_2)

        output2 = self.conv_2(dec1_2)

        return output1, output2

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )