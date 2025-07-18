import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint
    ct_idx = farthest_point_sample(xyz, npoint)
    torch.cuda.empty_cache()
    ct_xyz = index_points(xyz, ct_idx)
    torch.cuda.empty_cache()
    ft_idx = knn_point(nsample, xyz, ct_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, ft_idx)
    torch.cuda.empty_cache()
    ft_xyz = grouped_xyz - ct_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        ft_points = index_points(points, ft_idx)
        ct_points = index_points(points, ct_idx)
        ft_points = torch.cat([ft_xyz, ft_points], dim=-1)
    else:
        ft_points = ft_xyz
        ct_points = None

    return ct_xyz, grouped_xyz, ft_xyz, ft_points


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def relative_pos_encoding(nsample, ct_xyz, grouped_xyz):
    new_xyz = ct_xyz.unsqueeze(2)
    repeated_xyz = new_xyz.repeat(1, 1, nsample, 1)
    relative_xyz = grouped_xyz - repeated_xyz
    relative_dist = torch.sqrt(torch.sum(relative_xyz ** 2, dim=1, keepdim=True))
    relative_feature = torch.cat([relative_xyz, relative_dist], dim=1)
    return relative_feature


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        # 计算注意力权重的线性层
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=True),
            nn.ReLU(),  # 增加非线性激活使权重计算更复杂
            nn.Linear(in_channels, in_channels, bias=True),
            nn.Softmax(dim=-2)  # 保持 softmax 来计算注意力权重
        )

        # 增加多层卷积和 BatchNorm 提高特征的表达能力
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # 计算注意力权重
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 使用最大池化代替平均池化
        features = torch.max(scores * x, dim=-1, keepdim=True)[0]

        # 通过多层卷积和 BN 处理特征
        f_out = self.mlp(features)

        return f_out


class RandlaNet(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, d_in, d_out):
        super(RandlaNet, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.MLP_convs = nn.ModuleList()
        self.MLP_bns = nn.ModuleList()
        last_channel = in_channel + 3

        for out_channel in mlp:
            self.MLP_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.MLP_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.d_in = d_in
        self.d_out = d_out
        self.conv1 = nn.Conv2d(4, d_out // 2, 1)
        self.bn1 = nn.BatchNorm2d(d_out // 2)
        self.conv2 = nn.Conv2d(d_in, d_out // 2, 1)
        self.bn2 = nn.BatchNorm2d(d_out // 2)
        self.pool = AttentivePooling(d_out, 2 * d_out)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        ct_xyz, grouped_xyz, ft_xyz, ft_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        ct_xyz = ct_xyz.permute(0, 2, 1)
        grouped_xyz = grouped_xyz.permute(0, 3, 2, 1)
        ft_xyz = ft_xyz.permute(0, 3, 2, 1)
        ft_points = ft_points.permute(0, 3, 2, 1)

        #print(f"Initial ft_points shape: {ft_points.shape}")

        new_points1 = ft_points

        for i, conv in enumerate(self.MLP_convs):
            bn = self.MLP_bns[i]
            new_points1 = F.relu(bn(conv(new_points1)))
            #print(f"Shape after MLP layer {i+1}: {new_points1.shape}")

        new_points1 = torch.max(new_points1, 2)[0]
        #print(f"Shape after max pooling: {new_points1.shape}")

        new_xyz = relative_pos_encoding(self.nsample, ct_xyz, grouped_xyz)
        new_xyz = F.relu(self.bn1(self.conv1(new_xyz)))
        #print(f"Shape after relative position encoding: {new_xyz.shape}")

        new_points = F.relu(self.bn2(self.conv2(ft_points)))
        #print(f"Shape after convolution: {new_points.shape}")

        f_concat = torch.cat([new_xyz, new_points], dim=1)
        avg_pool = torch.mean(f_concat, 2)
        max_pool = torch.max(f_concat, 2)[0]
        new_points2 = torch.cat((avg_pool, max_pool), dim=1)

        conv_points = torch.cat((new_points1, new_points2), dim=1)
        #print(f"Final concatenated points shape: {conv_points.shape}")

        return ct_xyz, conv_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)

            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
