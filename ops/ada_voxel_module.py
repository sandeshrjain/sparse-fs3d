import torch
import torch.nn as nn
from .voxel_op import hard_voxelize


class AdaptiveVoxelizationGPU(nn.Module):
    def __init__(self, max_voxels_dense=12000, max_voxels_sparse=6000):
        super(AdaptiveVoxelizationGPU, self).__init__()
        self.dense_threshold = torch.tensor(25.0)
        self.voxel_size_dense_xy = torch.tensor([0.16, 0.16])
        self.voxel_size_sparse_xy = torch.tensor([0.32, 0.32])
        self.fixed_z = 4.0
        self.max_voxels_dense = max_voxels_dense
        self.max_voxels_sparse = max_voxels_sparse

    @property
    def voxel_size_dense(self):
        """Generate dense voxel size with fixed z."""
        return torch.cat(
            [self.voxel_size_dense_xy, torch.tensor([self.fixed_z], device=self.voxel_size_dense_xy.device)]
        )

    @property
    def voxel_size_sparse(self):
        """Generate sparse voxel size with fixed z."""
        return torch.cat(
            [self.voxel_size_sparse_xy, torch.tensor([self.fixed_z], device=self.voxel_size_sparse_xy.device)]
        )

    def voxelize(self, points, voxel_size, point_cloud_range, max_points=35, max_voxels=20000):
        if len(points) == 0:
            return None, None, None

        # Convert voxel_size to List[float]
        voxel_size = [float(v) for v in voxel_size.tolist()]

        # Preallocate tensors for output
        voxels_out = points.new_zeros(size=(max_voxels, max_points, points.size(1)), dtype=torch.float32)
        coors_out = points.new_zeros(size=(max_voxels, 3), dtype=torch.int32)
        num_points_per_voxel_out = points.new_zeros(size=(max_voxels,), dtype=torch.int32)

        # Invoke hard_voxelize
        voxel_num = hard_voxelize(
            points,
            voxels_out,
            coors_out,
            num_points_per_voxel_out,
            voxel_size,
            point_cloud_range,
            max_points,
            max_voxels,
            3,
            True,
        )

        # Process valid voxels
        voxels_out = voxels_out[:voxel_num]
        coors_out = coors_out[:voxel_num].flip(-1)  # Flip (z, y, x) -> (x, y, z)
        num_points_per_voxel_out = num_points_per_voxel_out[:voxel_num]

        return voxels_out, coors_out, num_points_per_voxel_out

    def forward(self, points, point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1], ada=True):
        """
        Forward pass toggles between adaptive and regular voxelization modes.
        Args:
            points: Input point cloud (N, ndim).
            point_cloud_range: Point cloud range (6D).
            ada: Whether to use adaptive voxelization (True) or regular (False).
        """
        if ada:
            # Perform adaptive voxelization
            voxels_dense, coors_dense, num_points_dense = self.voxelize(
                points, self.voxel_size_dense, point_cloud_range,
                max_points=35, max_voxels=self.max_voxels_dense
            )

            voxels_sparse, coors_sparse, num_points_sparse = self.voxelize(
                points, self.voxel_size_sparse, point_cloud_range,
                max_points=35, max_voxels=self.max_voxels_sparse
            )

            return {
                "dense": {
                    "voxels": voxels_dense,
                    "coors": coors_dense,
                    "num_points": num_points_dense
                },
                "sparse": {
                    "voxels": voxels_sparse,
                    "coors": coors_sparse,
                    "num_points": num_points_sparse
                }
            }
        else:
            # Perform regular voxelization
            return _Voxelization.apply(
                points, [0.16, 0.16, 4], point_cloud_range, 35, 20000, True
            )


class _Voxelization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True):
        """Hard voxelization forward function."""
        voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)
        voxel_num = hard_voxelize(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, 3, deterministic)

        # Select the valid voxels
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num].flip(-1)  # Flip from (z, y, x) to (x, y, z)
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        return voxels_out, coors_out, num_points_per_voxel_out


class Voxelization(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels, deterministic=True):
        super(Voxelization, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.deterministic = deterministic

    def forward(self, points):
        max_voxels = self.max_voxels[0] if self.training else self.max_voxels[1]
        return _Voxelization.apply(points, self.voxel_size, self.point_cloud_range, self.max_num_points, max_voxels, self.deterministic)

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, point_cloud_range={self.point_cloud_range}, max_num_points={self.max_num_points}, max_voxels={self.max_voxels}, deterministic={self.deterministic})"
