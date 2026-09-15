import torch
import torch.nn as nn


class PlacementPredictor(nn.Module):
    def __init__(
        self,
        num_shot_classes=6,
        position_dim=2,
        velocity_dim=2,
        pose_dim=34,
        grid_rows=3,
        grid_cols=4,
        hidden=(128, 64),
        dropout=0.2,
    ):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.pose_dim = pose_dim
        in_features = num_shot_classes + position_dim + velocity_dim + pose_dim

        layers = []
        for h in hidden:
            layers += [nn.Linear(in_features, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_features = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_features, grid_rows * grid_cols)

    def forward(self, shot_probs, hitter_pos, ball_vel, pose=None):
        parts = [shot_probs, hitter_pos, ball_vel]
        if self.pose_dim:
            parts.append(pose.flatten(1))
        return self.head(self.body(torch.cat(parts, dim=1)))

    @torch.no_grad()
    def heatmap(self, shot_probs, hitter_pos, ball_vel, pose=None):
        logits = self.forward(shot_probs, hitter_pos, ball_vel, pose)
        return logits.softmax(dim=1).view(-1, self.grid_rows, self.grid_cols)
