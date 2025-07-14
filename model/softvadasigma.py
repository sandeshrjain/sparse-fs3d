import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.anchors import Anchors, anchor_target, anchors2bboxes # Ensure these paths are correct
from ops import Voxelization, nms_cuda                 # Ensure these paths are correct
from utils import limit_period                         # Ensure this path is correct
from torch.utils.checkpoint import checkpoint



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.anchors import Anchors, anchor_target, anchors2bboxes
from ops import Voxelization, nms_cuda
from utils import limit_period
# Make sure checkpoint is imported if used in Backbone
from torch.utils.checkpoint import checkpoint
import math # For log

# Renamed to reflect the adaptive sigma change
class SoftPillarLayerAdaptiveSigma(nn.Module):
    """
    Applies Voxelization and Gaussian weighting with sigma predicted
    per-voxel based on point density (nums).
    Adds batch indices to coordinates.
    """
    def __init__(self,
                 # Revert to standard voxel size and max_voxels
                 voxel_size=(0.16, 0.16, 4.0),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 max_num_points=35,
                 max_voxels=(16000, 402.3): # Log(0.1) approx -2.3 - Base value for sigma predictor
        super().__init__()
        self.voxelizer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels
        )
        self.register_buffer('voxel_size', torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer('pc_range_min', torch.tensor(point_cloud_range[:3], dtype=torch.float32))
        self.max_num_points = float(max_num_points) # Ensure float for normalization

        # --- Sigma Predictor MLP000), # Standard values
                 initial_base_sigma=0.1, # Initial sigma value to center predictions around
                 sigma_mlp_hidden_dim=16): # Hidden dim for the small MLP
        super().__init__()
        self.voxelizer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels
        )
        self.register_buffer('voxel_size', torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer('pc_range_min', torch.tensor(point_cloud_range[:3], dtype=torch.float32))
        self.max_num_points = float(max_num_points) # Ensure float for normalization

        # --- Adaptive Sigma Components ---
        # Store ---
        # Predicts log_sigma_offset based on normalized point density
        # Input: Normalized point count (1 feature)
        # Output: Log sigma offset (1 feature)
        mlp_hidden_dim = 16 # Small hidden layer
        self.sigma_predictor_mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, base sigma in log space for stability when using exp
        self.register_buffer('base_log_sigma', torch.tensor(math.log(initial_base_sigma), dtype=torch.float32))

        # Small MLP to predict sigma offset based on normalized point count
        # Input: (M, 1) [normalized count], Output: (M, 1) [delta_log_sigma]
        self.sigma_predictor_mlp = nn.Sequential(
            nn.Linear(1 1)
            # Output is log_sigma_offset, we'll add base_log_sigma and exponentiate
        )
        # Register base log sigma as a non-trainable buffer
        self.register_buffer(', sigma_mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(sigma_mlp_hidden_base_log_sigma', torch.tensor(initial_base_log_sigma, dtype=torch.float3dim, 1)
        )
        # Initialize last layer bias to predict near zero initially,
        # so predicted2))
        # Initialize MLP weights (optional but can help)
        self._initialize_mlp_weights() sigma starts close to base_sigma
        nn.init.zeros_(self.sigma_predictor_mlp[-1].bias)


    def _initialize_mlp_weights(self):
        for m in self.sigma_predictor_mlp.modules():
        # --- End Adaptive Sigma Components ---

        # REMOVED: self.sigma = nn.Parameter(...)

    def forward(self, batched_pts):
        all_voxels_weighted, all_coors_            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(mbatch, all_nums_list = [], [], []

        for b_idx, pts in enumerate(batched_pts):
            .bias, 0)

    def forward(self, batched_pts):
        all_voxels_# 1. Voxelize
            voxels, coors, nums = self.voxelizer(pts)weighted, all_coors_batch, all_nums_cat = [], [], []

        # --- Step 1: Voxelize all # voxels(N_vox, P, C), coors(N_vox, 3), nums(N_vox,)
            if voxels.numel() == 0:
                continue

            device = voxels.device
            N_vox_sample = nums.shape[0]

            # 2. Predict Sigma per Voxel
            with torch.cuda points in the batch ---
        # Collect raw outputs first to process nums together
        temp_outputs = []
        for b.amp.autocast(enabled=False): # Calculate features in FP32 for stability if needed
                 _idx, pts in enumerate(batched_pts):
            voxels, coors, nums = self.voxelizer(pts)
            if voxels.numel() == 0:
                continue
            # Pad coors with batch index# Normalize point count feature
                 density_feature = (nums.float() / self.max_num_points).unsqueeze(1 *before* concatenation
            coors_with_batch = F.pad(coors.long(), (1, 0),) # Shape (N_vox_sample, 1)
                 # Ensure feature is float32 before MLP mode='constant', value=b_idx)
            temp_outputs.append({'voxels': voxels, ' if MLP weights are float32
                 density_feature = density_feature.float()

            # Predict offsetcoors': coors_with_batch, 'nums': nums})

        if not temp_outputs: # Handle empty batch
             return (
                 torch.zeros(0, int(self.max_num_points), 4, device=self.voxel_size.device),
                 torch.zeros(0, 4, dtype=torch.int32, device=self.voxel_size.device),
                 torch.zeros(0, dtype=torch.int32, device=self.voxel_size.device)
             )

        # Concatenate results across batch dimension in log-sigma space
            # Ensure MLP runs in appropriate precision (should match weights, typically float32 unless *before* adaptive sigma calculation
        voxels_cat = torch.cat([o['voxels'] for o in temp model converted)
            delta_log_sigma = self.sigma_predictor_mlp(density_feature) # Shape (_outputs], dim=0) # (M, P, C)
        coors_cat  = torch.cat([o['coN_vox_sample, 1)

            # Calculate final sigma per voxel: sigma = exp(base_log_sigma + deltaors'] for o in temp_outputs], dim=0)   # (M, 4) [b, z, y, x]_log_sigma)
            predicted_sigma = torch.exp(self.base_log_sigma + delta_log_sigma
        nums_cat   = torch.cat([o['nums'] for o in temp_outputs], dim=0)    # (M) # Shape (N_vox_sample, 1)
            # Clamp sigma to avoid instability
            sigma_clamped = predicted,)
        M = nums_cat.shape[0] # Total number of non-empty voxels in batch

        # --- Step 2_sigma.clamp(min=1e-4, max=10.0) # Add max clamp?: Predict Sigma per Voxel ---
        device = voxels_cat.device
        # Normalize point counts ( Clamp min tighter? (Experiment)


            # 3. Calculate Gaussian Weights
            centers = (coors.float() + 0.5) * self.voxel_size + self.pc_range_min # (density feature)
        density_feature = (nums_cat.float() / self.max_num_points).unsqueezeN_vox_sample, 3)
            P = voxels.shape[1]

            # Calculate distance(1) # (M, 1)

        # Predict log sigma offset using MLP
        log_sigma_offset =: (N_vox_sample, P)
            dist = torch.norm(voxels[..., :3] self.sigma_predictor_mlp(density_feature) # (M, 1)

        # Calculate predicted sigma per voxel
        predicted_ - centers.unsqueeze(1), dim=2)

            # Create mask for valid points: (N_vox_sample, Plog_sigma = self.base_log_sigma + log_sigma_offset # Add base value
        predicted_sigma = torch.)
            idx = torch.arange(P, device=device).view(1, -1)
            mask = (idxexp(predicted_log_sigma) # Convert log sigma to sigma (M, 1)

        # Clamp sigma to ensure < nums.unsqueeze(1)).float()

            # Calculate weights using per-voxel sigma: (N_vox_sample, P)
            # sigma_clamped has shape (N_vox_sample, 1), broadcasts minimum value
        sigma_clamped = predicted_sigma.clamp(min=1e-4) # ( correctly with dist
            w = torch.exp(-0.5 * (dist / sigma_clamped)**2) * mask
            wsum = w.sum(dim=1, keepdim=True).clamp(min=1e-6M, 1) - increased min slightly

        # --- Step 3: Apply Gaussian Weighting using predicted sigma ---
        # Calculate)
            w_normalized = w / wsum # Shape (N_vox_sample, P)

            # 4. voxel centers
        centers = (coors_cat[:, 1:].float() + 0.5) * self. Apply weights
            # Ensure multiplication happens in appropriate precision
            voxels_weighted = voxels * w_normalized.unsqueeze(-1)voxel_size.view(1, 3) + self.pc_range_min.view(1, 3) # ( # (N_vox_sample, P, C)

            # 5. Pad coordinates with batch index
            coors_M, 3), use coors z,y,x
        # Calculate distances
        P = int(self.max_num_points)
        dist = torch.norm(voxels_cat[..., :3]with_batch = F.pad(coors.long(), (1, 0), mode='constant', value=b_idx) # (N_vox_sample, 4) [b, z, y, x]

            # Collect results for - centers.unsqueeze(1), dim=2) # (M, P)
        # Create point mask
        idx = torch. this sample
            all_voxels_weighted.append(voxels_weighted)
            all_coors_batch.append(coors_with_batch)
            all_nums_list.append(nums)


arange(P, device=device).view(1, -1)
        mask = (idx < nums_        # Handle cases where the entire batch might be empty
        if not all_voxels_weighted:
             cat.unsqueeze(1)).float() # (M, P)

        # Calculate weights using the *per-voxel* sigma
        # Ensure sigma_clamped broadcasts correctly (M, 1) with dist (M, P)
# Return empty tensors with correct expected shapes
             # Determine expected feature dim C (usually 4 for x,y,z,        w = torch.exp(-0.5 * (dist / sigma_clamped)**2) * mask # (M, P)
        wsum = w.sum(dim=1, keepdim=True).clamp(min=1e-6) # (M, 1)
        w_normalized = w / wsum # (M, P)

        # Applyr)
             # Use a default C=4 if no weighted voxels were produced
             feature_dim = all_voxels_weighted[0].shape[-1] if all_voxels_weighted else 4
             return (
                 torch.zeros(0, int(self.max_num_points), feature_dim, device=self weights to original voxel features
        # Unsqueeze w_normalized to broadcast across feature dimension C
        voxels_weighted.voxel_size.device),
                 torch.zeros(0, 4, dtype=torch.int32, device=self.voxel_size.device),
                 torch.zeros(0, dtype=torch. = voxels_cat * w_normalized.unsqueeze(-1) # (M, P, C)

        # Returnint32, device=self.voxel_size.device)
             )

        # Concatenate results across the batch dimension
        pillars = torch.cat(all_voxels_weighted, dim=0) # (M_total, P, C)
        coors   = torch.cat(all_coors_batch, dim= the final weighted pillars, coordinates with batch index, and original nums
        return voxels_weighted, coors_cat, nums0)   # (M_total, 4) -> [b, z, y, x]
        nums    = torch._cat


# --- PillarEncoder remains unchanged from your previous working version ---
class PillarEncoder(nn.Module):
    defcat(all_nums_list, dim=0)    # (M_total,)

        return pillars, __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
         coors, nums


# --- PillarEncoder remains the same as your previous working version ---
class PillarEncoder(nnvx, vy, _ = voxel_size
        x0, y0, z0, x1, y1, z1 = point_cloud_range
        self.vx, self.vy = vx, vy
        self.x_.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        vx, vy, _ = voxel_size
        x0, y0, z0, x1, y1, z1 = point_cloud_range
        selfoffset = self.vx / 2 + x0
        self.y_offset = self.vy / 2 + y0
.vx, self.vy = vx, vy
        self.x_offset = self.vx / 2 + x0        self.x_l = int(np.round((x1 - x0) / vx))
        self.y_l = int(np.round((y1 - y0) / vy))
        self.conv = nn.Conv
        self.y_offset = self.vy / 2 + y0
        self.x_l = int(np.round((x1 - x0) / vx))
        self.y_l = int(np.round((1d(in_channel, out_channel, 1, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01y1 - y0) / vy))

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel, eps=1e-)

    def forward(self, pillars, coors, nums):
        # pillars: (M, P, C3, momentum=0.01)

    def forward(self, pillars, coors, nums):
        device = pillars.device
        M, P, C_in = pillars.shape

        # --- Feature Augmentation_in=4), coors: (M, 4) [b, z, y, x], nums: (M,)
        device = pillars.device
        M, P, C_in = pillars.shape

        # Feature Augmentation
 ---
        point_mask_p = (torch.arange(P, device=device).unsqueeze(0) < nums.view(-1, 1)).float() # (M, P)
        masked_pillars_        point_mask_p3 = (torch.arange(P, device=device).unsqueeze(0).unsqueeze(-1) < nums.viewxyz = pillars[:, :, :3] * point_mask_p.unsqueeze(-1)
        mean = masked_pillars_xyz(-1, 1, 1)).float() # (M, P, 1)
        masked_pillars_xyz = pillars[:, :, :3] * point_mask_p3 # (M, P, 3).sum(dim=1, keepdim=True) / nums.float().view(-1, 1, 1).clamp(min=1.0)
        f_centered = pillars[:, :, :3] - mean

        y_coor_grid = coors[:, 2].float()
        x_coor_grid = coors[:, 3].float()
        x_center_pillar = x_coor_grid * self.vx + self.x
        mean = masked_pillars_xyz.sum(dim=1, keepdim=True) / nums.float().view(-1, 1, 1).clamp(min=1.0) # (M, 1, 3)
        f_offset
        y_center_pillar = y_coor_grid * self.vy + self.y_offset
        f_x = pillars[:, :, 0:1] - x_center_pillar.view(-1, 1, 1)_centered = pillars[:, :, :3] - mean

        y_coor_grid = coors[:, 2].float() #
        f_y = pillars[:, :, 1:2] - y_center_pillar.view(-1 Y grid index
        x_coor_grid = coors[:, 3].float() # X grid index
        x_center_pillar = x_coor_grid * self.vx + self.x_offset
        y_center_, 1, 1)

        # Assemble features: [x, y, z, r, dx_mean, dy_mean, dzpillar = y_coor_grid * self.vy + self.y_offset
        f_x = pillars_mean, dx_center, dy_center] -> 9 features
        if C_in != 4:
             print(f"Warning: PillarEncoder input pillars dimension C_in={C_in}, expected 4.")
             # Handle potential mismatch if weighting changed feature dim (unlikely here)
             pillars_base = pillars[:,:,[:, :, 0:1] - x_center_pillar.view(-1, 1, 1)
        f_y =:4] # Take first 4 features assuming they are x,y,z,r
        else:
             pillars pillars[:, :, 1:2] - y_center_pillar.view(-1, 1, 1_base = pillars

        features = torch.cat([pillars_base, f_centered, f_x, f_y)

        # Assemble features
        if C_in != 4: print(f"Warning: PillarEncoder Input C_in={C_in}, expected 4.")
        features = torch.cat([pillars, f_centered, f_x, f_y], dim=2) # (M, P, 9)
        features = features * point], dim=2) # (M, P, 4+3+1+1=9)

        features_mask_p3 # Apply mask (redundant if masked_pillars_xyz used?)

        # PFN Layer
        features = features.permute(0, 2, 1).contiguous()
        features = F.relu(self. = features * point_mask_p.unsqueeze(-1) # Mask again after concat

        # --- PFN Layer ---
        features = features.permute(0, 2, 1).contiguous() # (M, 9, P)
        features = F.relu(bn(self.conv(features)))
        pooled   = torch.max(features, dim=2)[0]

        # Scatter features
        batch_ids = coors[:, 0].long()
        B = int(batch_ids.max().item()) + 1 if M > 0 else 0 # Handle empty caseself.bn(self.conv(features)))   # (M, out_channel, P)
        pooled

        canvases = []
        for b in range(B):
            sel = (batch_ids == b)   = torch.max(features, dim=2)[0]        # (M, out_channel)

        # --- Scatter ---
        
            canvas = torch.zeros(self.conv.out_channels, self.y_l, self.x_lbatch_ids = coors[:, 0].long()
        # Handle potential empty batch_ids if input was, device=device, dtype=pooled.dtype)
            if torch.any(sel):
                coors_batch = coors[ empty
        if batch_ids.numel() == 0:
             # Need to determine B from context if possible,sel]
                feat_batch = pooled[sel]
                y_indices = coors_batch[:, 2].long()
                x_indices = coors_batch[:, 3].long()
                valid_mask = (x_indices >= 0) & (x_indices < self.x_l) & (y_indices >=  or default/handle error
             # If pillars was empty, M=0, batch_ids is empty.
             # We0) & (y_indices < self.y_l)
                if valid_mask.any():
                    canvas[:, y_indices[valid_mask], x_indices[valid_mask]] = feat_batch[valid should return an empty canvas matching expected output shape
             # Assuming B=1 if batch_ids is empty (might_mask].t()
            canvases.append(canvas)

        # Handle case where B=0 (empty batch resulted in no canvases)
        if not canvases:
             return torch.zeros(0, self.conv.out_channels need actual B from outside?)
             B = 1 # Placeholder, might need actual batch size if M=0
             print, self.y_l, self.x_l, device=device, dtype=pooled.dtype)

        return torch.stack(canvases, dim=0)


# --- Backbone with checkpointing remains unchanged ---
class Backbone(nn.("Warning: Empty batch_ids in PillarEncoder Scatter.")
             # Return shape needs to match B, out_channelModule):
    def __init__(self, in_channel, out_channels, layer_nums, strides=[2,2,2]):
, y_l, x_l
             # Get out_channel from self.conv
             out_channels =        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_channel
 self.conv.out_channels
             return torch.zeros((B, out_channels, self.y_l, self.x        for oc, ln, st in zip(out_channels, layer_nums, strides):
            seq = [
                nn.Conv2d(ch, oc, 3, stride=st, padding=1,_l), device=device, dtype=pooled.dtype)


        B = int(batch_ids.max bias=False),
                nn.BatchNorm2d(oc, eps=1e-3, momentum=0().item()) + 1
        out_channels = self.conv.out_channels # Get C from conv layer

.01),
                nn.ReLU(inplace=True)
            ]
            for _ in range(ln):
                seq += [
                    nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                    nn.        canvases = []
        for b in range(B):
            sel  = (batch_ids == b)BatchNorm2d(oc, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)
                ]
            self.blocks.append(nn.Sequential(*seq))

            canvas = torch.zeros(out_channels, self.y_l, self.x_l,            ch = oc

    def forward(self, x):
        outs = []
        use_reentrant device=device, dtype=pooled.dtype) # Create canvas first
            if not torch.any(sel):
                 canvases.append(canvas) # Append empty canvas
                 continue

            coors_batch = coors[_strategy = False
        for block in self.blocks:
            # Apply checkpointing only if input requires grad (sel]
            feat_batch = pooled[sel]
            y_indices = coors_batch[:, 2].long()
            x_indices = coors_batch[:, 3].long()

            valid_mask = (x_indices >= 0) & (x_indices < self.x_l) & \
                         (y_indices >= 0)saves overhead if not)
            x = checkpoint(block, x, use_reentrant=use_reentrant & (y_indices < self.y_l)

            if valid_mask.any():
                 canvas[:,_strategy, preserve_rng_state=True) if x.requires_grad else block(x)
            outs y_indices[valid_mask], x_indices[valid_mask]] = feat_batch[valid_mask.append(x)
        return outs

# --- Neck remains unchanged ---
class Neck(nn.Module):
    def __init__(].t()
            canvases.append(canvas)

        return torch.stack(canvases,self, in_channels, up_strides, out_channels):
        super().__init__()
        self.decoders = nn.Module dim=0)


# --- Backbone (with checkpointing) ---
class Backbone(nn.Module):
    # Revert to standard channel configuration
    def __init__(self, in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5], strides=[1, 2, 2]):List()
        for ic, us, oc in zip(in_channels, up_strides, out_channels):
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(ic, oc, us, stride=us, bias=False),
                nn.BatchNorm2d(oc, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ))

    def forward(self, feats # Note: First stride is often 1
        super().__init__()
        # Adjust strides: First block downsamples Pillar Feature Net (PFN) output,
        # subsequent blocks downsample feature maps.
        # E.g., P):
        ups = [dec(f) for dec,f in zip(self.decoders, feats)]
        return torch.cat(ups, dim=1)

# --- Head remains unchanged ---
class Head(nn.Module):
    def __init__(self, in_chan, n_anchors, n_classes):
        super().__init__()
        self.n_anchors = n_anchors
        self.conv_cls = nn.Conv2d(in_chan, self.n_anchors * n_classes, 1)
        self.conv_FN spatial size YxX, Backbone Block1 YxX -> Block2 Y/2 x X/2 -> Block3 Y/reg = nn.Conv2d(in_chan, self.n_anchors * 7, 1)
        self.conv_4 x X/4
        # If PFN output is ~496x432 (0.16 voxdir = nn.Conv2d(in_chan, self.n_anchors * 2, 1)
        self._initialize_weights() # Encapsulate init logic

    def _initialize_weights(self):
        layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv), strides=[1,2,2] matches common setups.
        # If PFN output is ~248x216 (2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    if layer_id == 0:0.32 vox), strides=[1,2,2] still makes sense.
        if len(str
                        b = -np.log((1-0.01)/0.01)
                        nn.init.ides) != len(out_channels):
             print(f"Warning: Length mismatch strides({len(strides)})constant_(m.bias, b)
                    else:
                        nn.init.constant_(m.bias, 0)
                layer vs out_channels({len(out_channels)}). Using default [1,2,2] pattern if possible_id += 1

    def forward(self, x):
        return self.conv_cls(x), self.conv_reg.")
             # Adjust strides based on out_channels length if needed, common pattern is [1, 2, ...,(x), self.conv_dir(x)


# --- Main PointPillars Class ---
class PointPillars(nn.Module):
    def __init__(self,
                 nclasses=3,
                 # Revert to smaller 2]
             strides = [1] + [2] * (len(out_channels) - 1) if voxel size for potentially better performance
                 voxel_size=(0.16, 0.16, 4), len(out_channels) > 0 else []


        self.blocks = nn.ModuleList()
        ch = in_channel
        for i, (oc, ln, st) in enumerate(zip(out_channels, layer_nums, strides)):
                 pc_range=(0, -39.68, -3, 69.12, 3
            seq = [
                nn.Conv2d(ch, oc, 3, stride=st, padding=9.68, 1),
                 max_points=35,
                 # Revert to larger1, bias=False),
                nn.BatchNorm2d(oc, eps=1e-3, momentum=0.01), max_voxels
                 max_voxels=(16000, 40000),
                nn.ReLU(inplace=True)
            ]
            for _ in range(ln -1): # Original
                 # Backbone / Neck channels (use standard higher capacity)
                 backbone_channels=[64, 128, 256],
                 neck_channels=[128, 128, 128],
                 initial had ln loops *after* the first conv+bn+relu
                 seq += [
                     nn.Conv2d(oc,_sigma=0.1): # Pass initial sigma to adaptive layer
        super().__init__()
        self.nclasses = nclasses oc, 3, padding=1, bias=False),
                     nn.BatchNorm2d(oc, eps=1e-3

        # --- Use the NEW Adaptive Sigma Pillar Layer ---
        self.pillar_layer = SoftPillarLayerAdaptiveSigma, momentum=0.01),
                     nn.ReLU(inplace=True)
                 ]
            #(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points Check if ln was meant to be total layers including the first strided one
            # Original code added ln loops *in=max_points,
            max_voxels=max_voxels,
            initial_base_log addition* to the strided conv block.
            # If layer_nums=[3,5,5] means total_sigma=np.log(initial_sigma) # Pass log sigma
        )
        # --- End Change ---

        # PillarEncoder input channel = 9 features
        self.pillar_encoder = PillarEncoder(voxel layers in block:
            # for _ in range(ln - 1): # Adjust loop count if ln includes_size, pc_range, in_channel=9, out_channel=backbone_channels[0])

        # Use standard higher capacity backbone/neck
        self.backbone = Backbone(backbone_channels[0], backbone strided layer
            # Sticking to original interpretation for now: strided conv + ln additional convs.


            _channels, layer_nums=[3, 5, 5]) # Example layer nums
        self.neck = Neck(backbone_channels, up_strides=[1, 2, 4], out_channels=self.blocks.append(nn.Sequential(*seq))
            ch = oc

    def forward(self, xneck_channels)

        # Head input channel is sum of neck output channels
        neck_out_channel_sum = sum(neck):
        outs = []
        use_reentrant_strategy = False
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=use_reentrant_strategy)
            _channels) # e.g., 128*3 = 384

        # Calculate nouts.append(x)
        return outs


# --- Neck ---
class Neck(nn.Module):
    #_anchors based on anchor generator setup
        # Anchor setup (assuming 3 sizes * 2 rotations)
 Revert to standard channel configuration matching Backbone output
    def __init__(self, in_channels=[64, 128, 256], up_strides=[1, 2, 4], out_channels=[        num_anchor_sizes = 3
        num_anchor_rots = 2
        # Head expects128, 128, 128]):
        super().__init__()
        self.decoders = nn.ModuleList()
        for ic, us, oc in zip(in_channels, up_strides, total anchors per location (across all classes if classes are in channel dim)
        self.n_anchors_total_per_ out_channels):
            self.decoders.append(nn.Sequential(
                # Kernel size should matchloc = num_anchor_sizes * num_anchor_rots # 6

        self.head = Head(neck_out_channel stride for non-overlapping transpose conv if desired
                nn.ConvTranspose2d(ic, oc, us,_sum, n_anchors=self.n_anchors_total_per_loc, n_classes=n stride=us, bias=False),
                nn.BatchNorm2d(oc, eps=1e-3, momentum=0.classes)

        # Anchor Generator Setup (same as before)
        ranges = [ # Class specific ranges (adjust if needed)
01),
                nn.ReLU(inplace=True)
            ))

    def forward(self, feats            [0,-39.68,-0.6, 69.12,39.68,-0.6],
            [0,-39.68,-0.6, 69.12,3):
        # Ensure feats list matches in_channels length
        if len(feats) != len(self.decoders):
9.68,-0.6],
            [0,-39.68,-1.78             print(f"Warning: Neck received {len(feats)} features, expected {len(self.decoders)}"),69.12,39.68,-1.78]
        ]
        sizes = [
             # Handle mismatch if possible, e.g. use subset of feats?
             # Assuming feats aligns correctly for now.

        ups = # Class specific sizes [w, l, h]
            [1.6, 3.9, 1.5 [dec(f) for dec,f in zip(self.decoders, feats)]
        return torch.cat(ups, dim=16], # Car
            [0.6, 0.8, 1.73], # Ped
            [)


# --- Head ---
class Head(nn.Module):
    # Adjust input channel to match Neck output sum
    def __init__(0.6, 1.76, 1.73] # Cyc
        ]
        rots = [0, npself, in_chan=384, n_anchors=6, n_classes=3): # n_anchors per.pi/2]
        self.anchors_generator = Anchors(ranges=ranges, sizes=sizes, rotations=rots)

        # Target Assigner Setup (same as before)
        self.assigners = [
            {'pos_iou_thr': 0. location = 6 default
        super().__init__()
        self.n_anchors = n_anchors

6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45}, #        self.conv_cls = nn.Conv2d(in_chan, self.n_anchors * n_classes, 1)
        self.conv_reg = nn.Conv2d(in_chan, self.n_anch Car
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35}, # Ped
            {'pos_ors * 7,       1)
        self.conv_dir = nn.Conv2d(in_chan, self.n_anchors * 2,       1)

        layer_id = 0
        for m in self.iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35}, # Cyc
        ]

        # Post-processing Paramsmodules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m (same as before)
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:num = 50

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)

        pillars, coors
                    if m is self.conv_cls: # Check specifically for the cls conv
                        b = -math.log((1 - 0.01) / 0.01) # Focal loss prior init
                        nn.init.constant, nums = self.pillar_layer(batched_pts)

        if pillars.numel() == 0 and_(m.bias, b)
                    else:
                        nn.init.constant_(m.bias, 0)
                layer mode == 'train':
             # Handle empty batch during training - need structure for target assigner
             print("Warning_id += 1 # Increment layer_id based on module traversal order? Or specific layers?

    def forward(self, x: Pillar layer produced empty output during training.")
             # Create dummy outputs that won't crash target assigner/):
        return self.conv_cls(x), self.conv_reg(x), self.conv_dir(x)


# --- Main PointPillars Module ---
class PointPillars(nn.Module):
    def __loss
             # Get expected output feature map size (difficult without running head)
             # A cleaner solution might beinit__(self,
                 nclasses=3,
                 # Revert to standard voxel size and max_voxels
                 voxel_size=(0.16, 0.16, 4),
                 pc_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 max_points needed in anchor_target/loss to handle this
             # For now, return empty dict for tgt? Check anchor=35,
                 max_voxels=(16000, 40000),
                 initial_target behavior.
             # Returning empty predictions might be safer:
             dummy_head_out_shape = (_base_sigma=0.1, # Pass initial sigma to adaptive layer
                 sigma_mlp_hidden_dim=16):batch_size, self.n_anchors_total_per_loc * self.nclasses, 1, 1 # Pass MLP dim to adaptive layer
        super().__init__()
        self.nclasses = nclasses

        # Use the Adaptive Sigma layer
        self.pillar_layer   = SoftPillarLayerAdaptiveSigma(
                                  voxel_size, pc_range, max_points, max_voxels,
                                  initial_base_sigma, sigma_mlp_hidden_dim
) # Minimal spatial size
             dummy_reg_shape = (batch_size, self.n_anchors_total_per_loc * 7, 1, 1)
             dummy_dir_shape = (batch_size, self.n_anchors_total_per_loc * 2, 1, 1)
             dummy_device = self.head                              )
        # PillarEncoder input channel = 9 features
        self.pillar_encoder = PillarEncoder(voxel_size, pc_range, in_channel=9, out_channel=64)

        # Re.conv_cls.weight.device # Get device from a parameter
             cls_p = torch.zeros(dummy_head_vert Backbone/Neck/Head to standard channels
        backbone_channels = [64, 128, 256]out_shape, device=dummy_device)
             reg_p = torch.zeros(dummy_reg_shape, device=dummy_device
        neck_channels     = [128, 128, 128]
        self)
             dir_p = torch.zeros(dummy_dir_shape, device=dummy_device)
             #.backbone     = Backbone(in_channel=64, out_channels=backbone_channels, layer_nums=[3, 5 Return dummy target dict expected by loss function
             tgt = {'batched_labels': torch.empty((batch_size, 0), dtype, 5])
        self.neck         = Neck(in_channels=backbone_channels, up_strides=[1, 2, 4], out_channels=neck_channels)
        head_in_channels  = sum(neck=torch.long, device=dummy_device),
                   'batched_label_weights': torch.empty((batch_size, 0_channels) # 128*3 = 384

        # Determine n_anchors per location), dtype=torch.float, device=dummy_device),
                   'batched_bbox_reg': torch.empty((batch_size, 0, 7), dtype=torch.float, device=dummy_device),
                   'batched_dir_labels': based on anchor generator
        # Common KITTI: 3 sizes * 2 rotations = 6 anchors per class torch.empty((batch_size, 0), dtype=torch.long, device=dummy_device)}
             return cls_p, reg_p, dir_p, tgt

        elif pillars.numel() == 0 and per location
        # Head needs total anchors per location if classes are in channels
        num_anchor_sizes = 3 # Example (mode=='val' or mode=='test'):
             print("Warning: Pillar layer produced empty output during eval/test.")
             # Car, Ped, Cyc
        num_anchor_rots = 2 # Example 0, 90 deg
        n Return empty results list
             return [{'lidar_bboxes': np.array([]), 'labels': np.array([]), 'scores': np_anchors_per_loc = num_anchor_sizes * num_anchor_rots # 6

        self.head = Head.array([])} for _ in range(batch_size)]


        canvas  = self.pillar_encoder(pillars, coors, nums(in_chan=head_in_channels, n_anchors=n_anchors_per_loc, n)
        feats   = self.backbone(canvas)
        fused   = self.neck(feats_classes=nclasses)

        # --- Anchor Generator Setup ---
        # Ensure these are suitable for voxel_size )
        cls_p, reg_p, dir_p = self.head(fused)

        # Mode handling0.16x0.16
        # Ranges and sizes seem reasonable for standard KITTI setup
        ranges = [
            
        if mode == 'train':
            fmap_h, fmap_w = cls_p.[0, -39.68, -0.6, 69.12, 3shape[-2:]
            device = cls_p.device
            feature_map_size = torch.tensor([fmap_h, fmap_w], device=device)
            anchors = self.anchors9.68, -0.6],
            [0, -39.68, -0.6, 69.12, 39.68, -0.6],
            [0, -39.68, -1.78, 69.12, _generator.get_multi_anchors(feature_map_size)
            batched_anchors = [anchors.clone() for _ in range(batch_size)]

            tgt = anchor_target(batched_anchors=batched_39.68, -1.78]
        ]
        sizes  = [
            [1.6anchors,
                                batched_gt_bboxes=batched_gt_bboxes,
                                batched_gt_labels=batched_gt_labels,
                                assigners=self.assigners,
                                nclasses=self.nclasses)
            return cls_p, 3.9, 1.56], # Car size
            [0.6, 0.8, 1.73], # Pedestrian size
            [0.6, 1.76, 1.73] # Cyclist size
        ]
        rots   = [0, math.pi / 2]

        self.anchors_generator = Anchors, reg_p, dir_p, tgt

        elif mode == 'val' or mode == 'test':
            fmap_h, fmap_w = cls_p.shape[-2:]
            device = cls_p.device
            feature_map_(ranges=ranges, sizes=sizes, rotations=rots)

        # --- Target Assigner Setup ---
        self.assignsize = torch.tensor([fmap_h, fmap_w], device=device)
            anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
            batched_anchors = [anchers  = [
            {'pos_iou_thr': 0.6, 'neg_iou_thr': ors.clone() for _ in range(batch_size)]
            outs = []
            for i in range(batch_size):
                 decoded = self._decode_single(cls_p[i], reg_p[i], dir_p0.45, 'min_iou_thr': 0.45}, # Car
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou[i], batched_anchors[i])
                 outs.append(decoded)
            return outs
        else:
             raise ValueError(f"Unknown mode: {mode}")


    # _decode_single remains_thr': 0.35}, # Ped
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0. unchanged from your previous working version
    def _decode_single(self, c_p, r_p, d35}, # Cyc
        ]

        # --- Post-processing Params ---
        self.nms_pre    = 100 # Max boxes before NMS (per class)
        self.nms_thr    = 0.0_p, anchors):
        num_anchors = self.n_anchors_total_per_loc
        1 # NMS threshold
        self.score_thr  = 0.1 # Score threshold
        self.max_num    = 50 # Max final boxes

    def forward(self, batched_pts, mode='test', batbbox_cls = c_p.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_reg = r_p.permute(1, 2, 0).reshape(-1, 7)ched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)

        # --- Pillar Encoding ---
        pillars, coors, nums = self.pillar_layer(
        dir_cls  = d_p.permute(1, 2, 0).reshape(-1, 2)
        anchors_flat = anchors.reshape(-1, 7)
        bbox_cls = bbox_cls.sigmoid()
batched_pts)
        # Handle empty case immediately if possible
        if pillars.numel() == 0:
                     dir_cls  = torch.argmax(dir_cls, dim=1)

        top_scores, _ = bbox_cls.max(dim=1)
        if top_scores.numel() == 0print("Warning: Pillar layer produced empty output. Returning empty structure.")
             # Determine expected output shapes for downstream processing:
             return {'lidar_bboxes': np.array([]), 'labels': np.array([]), 'scores': np.array([])
             # Need C_cls, C_reg, C_dir, H_out, W_out
}
        nms_pre_count = min(self.nms_pre, top_scores.numel())
        top_scores, top_inds = top_scores.topk(nms_pre_count)

        bbox_cls = bbox_cls[top_inds]
        bbox_reg = bbox_reg[top_inds]
        dir_cls               # Use dummy values or calculate from config if possible
             C_cls = self.head.conv_cls.out_channels
             C_reg = self.head.conv_reg.out_channels
             C_dir = self.= dir_cls[top_inds]
        anchors_flat = anchors_flat[top_inds]

        bbox3d = anchors2bboxes(anchors_flat, bbox_reg)
        xy_centerhead.conv_dir.out_channels
             # Estimating H/W is harder without canvas - maybe return = bbox3d[:, [0, 1]]
        wl = bbox3d[:, [3, 4]]
        bev_boxes_xyxy = torch.cat([xy_center - wl / 2, xy_center + wl / 2], None or raise?
             # Returning empty tensors might be best if loss/eval can handle it.
             dummy dim=1)

        final_bboxes, final_labels, final_scores = [], [], []
        for cls_id in range(self.nclasses):
            class_scores = bbox_cls[:, cls_id]
            score__h, dummy_w = 1, 1 # Placeholder
             empty_cls = torch.zeros((batchmask = class_scores > self.score_thr
            if not score_mask.any(): continue

            scores_C = class__size, C_cls, dummy_h, dummy_w), device=self.head.conv_cls.weight.device)
scores[score_mask]
            boxes_bev_C = bev_boxes_xyxy[score_mask]
            boxes_3d             empty_reg = torch.zeros((batch_size, C_reg, dummy_h, dummy_w_C = bbox3d[score_mask]
            dir_C = dir_cls[score_mask]

            keep_indices = nms_cuda(boxes=boxes_bev_C, scores=scores_C, thresh=self), device=empty_cls.device)
             empty_dir = torch.zeros((batch_size, C.nms_thr)
            if keep_indices.numel() == 0: continue

            final__dir, dummy_h, dummy_w), device=empty_cls.device)
             empty_tgt = {'batched_labels': torch.empty((batch_size, 0), dtype=torch.long, device=emptybboxes_C = boxes_3d_C[keep_indices]
            final_scores_C = scores_C[keep_indices]_cls.device),
                          'batched_label_weights': torch.empty((batch_size, 0), dtype=
            final_dir_C = dir_C[keep_indices]

            yaw = final_bboxes_C[:, 6]
            dir_correction = (1 - final_dir_C.float()) * torch.pi
            correctedtorch.float, device=empty_cls.device),
                          'batched_bbox_reg': torch.empty((batch_size, 0, 7), dtype=torch.float, device=empty_cls.device),
                          'batched_dir_labels': torch.empty((batch_size, 0), dtype=torch.long, device=_yaw = yaw + dir_correction
            # Ensure limit_period handles tensor inputs correctly
            final_bboxes_Cempty_cls.device)}
             if mode == 'train':
                 return empty_cls, empty_reg, empty_dir[:, 6] = limit_period(corrected_yaw, offset=0.5, period=2 * torch.pi)

            final_bboxes.append(final_bboxes_C)
            final_labels.append(torch.full((len, empty_tgt
             else: # val/test
                 return [{'lidar_bboxes': np.array([]), 'labels': np(keep_indices),), cls_id, dtype=torch.long, device=final_scores_C.device))
            final.array([]), 'scores': np.array([])} for _ in range(batch_size)]

        canvas  _scores.append(final_scores_C)

        if not final_bboxes:
             return {'lidar_bboxes':= self.pillar_encoder(pillars, coors, nums)
        # --- Backbone -> Neck -> Head ---
        feats   = self.backbone(canvas)
        fused   = self.neck(feats)
        cls_p, reg_p, dir np.array([]), 'labels': np.array([]), 'scores': np.array([])}

        final_bboxes = torch.cat(final_bboxes, dim=0)
        final_labels = torch.cat_p = self.head(fused)

        # --- Mode handling ---
        if mode == 'train':
            fmap_h, fmap_w = cls_p.shape[-2:]
            device(final_labels, dim=0)
        final_scores = torch.cat(final_scores, dim = cls_p.device
            feature_map_size = torch.tensor([fmap_h, fmap_w], device=device)
            anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
            =0)

        if final_bboxes.shape[0] > self.max_num:
            topk_scores, topk_indices = final_scores.topk(self.max_num)
            final_bboxes = final_bboxes[topk_indices]
batched_anchors = [anchors.clone() for _ in range(batch_size)]
            tgt            final_labels = final_labels[topk_indices]
            final_scores = topk_scores = anchor_target(batched_anchors=batched_anchors,
                                batched_gt_bboxes=batched_gt_bboxes,
                                batched_gt_labels=batched_gt_labels,
                                assigners=

        return {
            'lidar_bboxes': final_bboxes.detach().cpu().numpy(),
            'labels':       final_labels.detach().cpu().numpy(),
            'scores':       final_scores.detachself.assigners,
                                nclasses=self.nclasses)
            return cls_p, reg_().cpu().numpy()
        }