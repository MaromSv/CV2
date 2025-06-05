import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import BasicConv, ResBlock
from thop import profile


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True, norm=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=True)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True, norm=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True, norm=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True, norm=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True, norm=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False, norm=True)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False, norm=True)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOUNet(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1, norm=True),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2, norm=True),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2, norm=True),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True, norm=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True, norm=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1, norm=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1, norm=True),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = []

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        # z = self.Decoder[0](z)
        # z_ = self.ConvsOut[0](z)
        # z = self.feat_extract[3](z)
        # outputs.append(z_)
        # low-res head → 3 channels
        z = self.Decoder[0](z)
        out3 = self.feat_extract[3](z)      # [B, C3, H/4, W/4]
        tmp3 = self.ConvsOut[0](out3)       # [B,3,H/4,W/4]
        tmp3[:, 2:3] = torch.sigmoid(tmp3[:, 2:3])
        outputs.append(tmp3)

        # z = torch.cat([z, res2], dim=1)
        # z = self.Convs[0](z)
        # z = self.Decoder[1](z)
        # z_ = self.ConvsOut[1](z)
        # z = self.feat_extract[4](z)
        # outputs.append(z_)
        # mid-res head → 3 channels
        z = torch.cat([out3, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        out2 = self.feat_extract[4](z)      # [B,C3,H/2,W/2]
        tmp2 = self.ConvsOut[1](out2)       # [B,3,H/2,W/2]
        tmp2[:, 2:3] = torch.sigmoid(tmp2[:, 2:3])
        outputs.append(tmp2)

        # z = torch.cat([z, res1], dim=1)
        # z = self.Convs[1](z)
        # z = self.Decoder[2](z)
        # z = self.feat_extract[5](z)
        # outputs.append(z)
        # full-res head → 3 channels
        z = torch.cat([out2, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        tmp1 = self.feat_extract[5](z)      # [B,3,H,W]
        tmp1[:, 2:3] = torch.sigmoid(tmp1[:, 2:3])
        outputs.append(tmp1)

        return outputs


class MIMOUNetPlus(nn.Module):
    def __init__(self, num_res = 20):
        super(MIMOUNetPlus, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1, norm=True),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2, norm=True),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2, norm=True),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True, norm=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True, norm=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1, norm=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1, norm=True),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        # --- First output (lowest resolution) ---
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z_[:, 2:3] = torch.sigmoid(z_[:, 2:3])
        outputs.append(z_)
        z = self.feat_extract[3](z)

        # --- Second output (mid resolution) ---
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z_[:, 2:3] = torch.sigmoid(z_[:, 2:3])
        outputs.append(z_)
        z = self.feat_extract[4](z)

        # --- Final output (highest resolution) ---
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        z[:, 2:3] = torch.sigmoid(z[:, 2:3])
        outputs.append(z)

        return outputs

class DoubleConv(nn.Module):
    """(Conv => BatchNorm => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B, C, H, W]
        e2 = self.enc2(F.max_pool2d(e1, 2))  # [B, 2C, H/2, W/2]
        e3 = self.enc3(F.max_pool2d(e2, 2))  # [B, 4C, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))  # [B, 8C, H/8, W/8]

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out(d1)  # [B, out_channels, H, W]
        out[:, 2:3] = torch.sigmoid(out[:, 2:3])
        return out


def build_MIMOUnet_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-UNetPlus":
        return MIMOUNetPlus()
    elif model_name == "MIMO-UNet":
        return MIMOUNet()
    elif model_name == "UNet":
        return UNet()
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')

if __name__ == '__main__':
    # Debug
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = build_MIMOUnet_net("MIMO-UNet")
    net = net.cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(net, (input, ))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')