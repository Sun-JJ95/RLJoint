import torch.nn as nn
import torch.nn.functional as F
from new1 import RandlaNet, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.ra1 = RandlaNet(1024, radius=0.1, nsample=16, in_channel=3, mlp=[32, 32, 64], d_in=3 + 3, d_out=16)
        self.ra2 = RandlaNet(256, radius=0.2, nsample=16, in_channel=96, mlp=[64, 64, 128], d_in=96 + 3, d_out=32)
        self.ra3 = RandlaNet(64, radius=0.4, nsample=16, in_channel=192, mlp=[64, 128, 128], d_in=192 + 3, d_out=64)
        self.ra4 = RandlaNet(16, radius=0.8, nsample=16, in_channel=256, mlp=[128, 128, 256], d_in=256 + 3, d_out=128)

        self.fp4 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 192, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 96, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz[:, 3:6, :]
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.ra1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.ra2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.ra3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.ra4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l0_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss


if __name__ == '__main__':
    import torch

    model = get_model(13)
    xyz = torch.rand(4, 9, 4096)
    output, _ = model(xyz)
    print(output.shape)
