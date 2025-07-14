# ================================
# 1. Gaussian Soft Voxelization Module
# ================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianSoftVoxelization(nn.Module):
    def __init__(self, voxel_size=(0.16, 0.16, 4.0), point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_points_per_voxel=35, max_voxels=20000, sigma=0.1):
        super(GaussianSoftVoxelization, self).__init__()
        self.voxel_size = torch.tensor(voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))

    def forward(self, points):
        device = points.device
        voxel_size = self.voxel_size.to(device)
        point_cloud_range = self.point_cloud_range.to(device)

        voxel_indices = ((points[:, :3] - point_cloud_range[:3]) / voxel_size).floor().long()
        valid_mask = ((voxel_indices >= 0) & (voxel_indices < ((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).long())).all(dim=1)
        points = points[valid_mask]
        voxel_indices = voxel_indices[valid_mask]

        hash_indices = voxel_indices[:, 0] * 1000000 + voxel_indices[:, 1] * 1000 + voxel_indices[:, 2]
        unique_hashes, inverse_indices = torch.unique(hash_indices, return_inverse=True)

        voxel_coords = []
        voxel_features = []
        num_points_per_voxel = []

        for i in range(len(unique_hashes)):
            voxel_mask = (inverse_indices == i)
            pts_in_voxel = points[voxel_mask]
            coord = voxel_indices[voxel_mask][0]
            voxel_center = (coord.float() + 0.5) * voxel_size + point_cloud_range[:3]

            dist = torch.norm(pts_in_voxel[:, :3] - voxel_center[None, :], dim=1)
            weights = torch.exp(-0.5 * (dist / self.sigma.clamp(min=1e-3)) ** 2)

            weights, top_idx = torch.topk(weights, min(len(weights), self.max_points_per_voxel), largest=True)
            pts_top = pts_in_voxel[top_idx]
            weights = weights / weights.sum()

            padded_voxel = torch.zeros((self.max_points_per_voxel, pts_top.shape[1]), device=device)
            padded_voxel[:pts_top.size(0)] = pts_top * weights[:, None]

            voxel_features.append(padded_voxel)
            voxel_coords.append(coord)
            num_points_per_voxel.append(len(pts_top))

            if len(voxel_features) >= self.max_voxels:
                break

        voxel_features = torch.stack(voxel_features, dim=0)  # (M, max_points, C)
        voxel_coords = torch.stack(voxel_coords, dim=0)      # (M, 3)
        num_points_per_voxel = torch.tensor(num_points_per_voxel, device=device)

        return voxel_features, voxel_coords, num_points_per_voxel


# ================================
# 2. PillarLayer for Soft Voxelization
# ================================
class SoftPillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.soft_voxelizer = GaussianSoftVoxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_num_points,
            max_voxels=max_voxels
        )

    @torch.no_grad()
    def forward(self, batched_pts):
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.soft_voxelizer(pts)
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        pillars = torch.cat(pillars, dim=0)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)

        return pillars, coors_batch, npoints_per_pillar