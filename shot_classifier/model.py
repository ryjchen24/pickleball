import numpy as np
import torch
import torch.nn as nn

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]
NUM_JOINTS = 17


def build_adjacency(num_joints=NUM_JOINTS, edges=COCO_EDGES):
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    neighbors = A.copy()
    degree = neighbors.sum(axis=0)
    degree[degree == 0] = 1.0
    neighbors = neighbors / degree
    return np.stack([np.eye(num_joints, dtype=np.float32), neighbors])


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_subsets):
        super().__init__()
        self.num_subsets = num_subsets
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

    def forward(self, x, A):
        n, _, t, v = x.shape
        x = self.conv(x).view(n, self.num_subsets, self.out_channels, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, A)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_subsets, stride=1, kernel_t=9, dropout=0.0, residual=True):
        super().__init__()
        pad = (kernel_t - 1) // 2
        self.gcn = GraphConv(in_channels, out_channels, num_subsets)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_t, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A))
        return self.relu(x + res)


class STGCN(nn.Module):
    def __init__(
        self,
        num_classes=6,
        in_channels=3,
        num_joints=NUM_JOINTS,
        edges=COCO_EDGES,
        layers=((64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1)),
        aux_dim=0,
        dropout=0.3,
        edge_importance=True,
    ):
        super().__init__()
        A = torch.tensor(build_adjacency(num_joints, edges))
        self.register_buffer("A", A)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        blocks = []
        c_in = in_channels
        for i, (c_out, stride) in enumerate(layers):
            blocks.append(STGCNBlock(c_in, c_out, A.shape[0], stride, dropout=dropout, residual=i > 0))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)

        if edge_importance:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones_like(A)) for _ in blocks])
        else:
            self.edge_importance = [1.0] * len(blocks)

        self.aux_dim = aux_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in + aux_dim, num_classes))

    def features(self, x):
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).reshape(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.A * importance)
        return x.mean(dim=(2, 3))

    def forward(self, x, aux=None):
        feats = self.features(x)
        if self.aux_dim:
            feats = torch.cat([feats, aux], dim=1)
        return self.head(feats)
