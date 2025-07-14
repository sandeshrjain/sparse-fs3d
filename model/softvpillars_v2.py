import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.anchors import Anchors, anchor_target, anchors2bboxes
from ops import Voxelization, nms_cuda
from utils import limit_period
# Make sure checkpoint is imported if used in Backbone
from torch.utils.checkpoint import checkpoint

class SoftPillarLayer(nn.Module):
    """
    Applies Voxelization and Gaussian weighting, adding batch indices to coordinates.
    """
    def __init__(self,
                 # Using your specified voxel size and reduced max_voxels
                 voxel_size=(0.16, 0.16, 4.0),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 max_num_points=35,
                 max_voxels=(16000, 40000),
                 sigma=0.1):
        super().__init__()
        self.voxelizer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels # Use tuple for max_voxels
        )
        self.register_buffer('voxel_size', torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer('pc_range_min', torch.tensor(point_cloud_range[:3], dtype=torch.float32))
        self.max_num_points = max_num_points
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32)) # Keep sigma trainable

    # Remove @torch.no_grad() if sigma needs to be trained
    # def forward(self, batched_pts): # Keep grads for sigma
    # If sigma fixed or checkpointed elsewhere, you can use no_grad here
    # For now, assume sigma is trainable and remove @torch.no_grad()
    def forward(self, batched_pts):
        all_voxels, all_coors, all_nums = [], [], []
        # --- Add batch index using enumerate ---
        for b_idx, pts in enumerate(batched_pts):
            voxels, coors, nums = self.voxelizer(pts)
            if voxels.numel() == 0:
                continue

            device = voxels.device
            centers = (coors.float() + 0.5) * self.voxel_size + self.pc_range_min

            P = self.max_num_points
            dist = torch.norm(voxels[..., :3] - centers.unsqueeze(1), dim=2)
            idx = torch.arange(P, device=device).view(1, -1)
            mask = (idx < nums.unsqueeze(1)).float()

            # Clamp sigma inside calculations where gradient needed
            sigma_clamped = self.sigma.clamp(min=1e-3)
            w = torch.exp(-0.5 * (dist / sigma_clamped)**2) * mask
            wsum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            w_normalized = w / wsum # Keep normalized w for potential grad flow

            # Apply weights
            voxels_weighted = voxels * w_normalized.unsqueeze(-1)

            # --- Pad coors with batch index ---
            # coors shape: (N_voxels_sample, 3) -> (N_voxels_sample, 4) where dim 0 is batch index
            coors_with_batch = F.pad(coors.long(), (1, 0), mode='constant', value=b_idx) # Pad dim 1 (columns) on the left

            all_voxels.append(voxels_weighted)
            all_coors.append(coors_with_batch)
            all_nums.append(nums)

        if not all_voxels:
            # Return empty tensors with correct expected shapes if possible
            # PillarEncoder expects coors (M, 4) now
             return (
                 torch.zeros(0, self.max_num_points, pillars.shape[-1] if 'pillars' in locals() else 4, device=self.voxel_size.device), # Match feature dim if available
                 torch.zeros(0, 4, dtype=torch.int32, device=self.voxel_size.device), # Coors now has 4 dims
                 torch.zeros(0, dtype=torch.int32, device=self.voxel_size.device)
             )

        # Concatenate across batch dimension
        pillars = torch.cat(all_voxels, dim=0) # (M_total, P, C)
        coors   = torch.cat(all_coors, dim=0)   # (M_total, 4) -> [b, z, y, x]
        nums    = torch.cat(all_nums, dim=0)    # (M_total,)

        return pillars, coors, nums

class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        # Using specified voxel size
        vx, vy, _ = voxel_size
        x0, y0, z0, x1, y1, z1 = point_cloud_range
        self.vx, self.vy = vx, vy
        # Calculate offsets based on the actual min range
        self.x_offset = self.vx / 2 + x0
        self.y_offset = self.vy / 2 + y0
        # Calculate grid size (ensure integer division or rounding consistency)
        self.x_l = int(np.round((x1 - x0) / vx)) # Use round for robustness
        self.y_l = int(np.round((y1 - y0) / vy))

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors, nums):
        # pillars: (M, P, 4), coors: (M, 4) [b, z, y, x], nums: (M,)
        device = pillars.device
        M, P, C_in = pillars.shape # C_in should be 4 initially

        # --- Feature Augmentation (Ensure correct indexing) ---
        # 1) offset to mean point in pillar
        # Need mask for sum, divide by actual nums, handle nums=0
        masked_pillars = pillars[:, :, :3] * (torch.arange(P, device=device).unsqueeze(0).unsqueeze(-1) < nums.view(-1, 1, 1)).float() # (M, P, 3)
        mean = masked_pillars.sum(dim=1, keepdim=True) / nums.float().view(-1, 1, 1).clamp(min=1.0) # (M, 1, 3)
        f_centered = pillars[:, :, :3] - mean # Broadcasting works

        # 2) offset to pillar center (use x, y coordinates from coors)
        # coors columns are [b, z, y, x] -> index 2 is y, index 3 is x
        y_coor_grid = coors[:, 2].float() # Y grid index
        x_coor_grid = coors[:, 3].float() # X grid index
        # Calculate pillar center coordinates in point space
        x_center_pillar = x_coor_grid * self.vx + self.x_offset
        y_center_pillar = y_coor_grid * self.vy + self.y_offset
        # Calculate offset of each point from its pillar's center
        f_x = pillars[:, :, 0:1] - x_center_pillar.view(-1, 1, 1) # Offset in X
        f_y = pillars[:, :, 1:2] - y_center_pillar.view(-1, 1, 1) # Offset in Y

        # assemble features: [x, y, z, r, dx_mean, dy_mean, dz_mean, dx_center, dy_center] -> 9 features
        # Note: Original pillars has 4 features (x, y, z, r)
        if C_in != 4:
             print(f"Warning: Input pillars dimension C_in={C_in}, expected 4.") # Debug print
        features = torch.cat([pillars, f_centered, f_x, f_y], dim=2) # (M, P, 4+3+1+1=9)

        # Mask out features for points beyond nums
        point_mask = (torch.arange(P, device=device).view(1, -1) < nums.view(-1, 1)).float().unsqueeze(2) # (M, P, 1)
        features = features * point_mask

        # --- PFN Layer ---
        features = features.permute(0, 2, 1).contiguous()           # (M, 9, P)
        features = F.relu(self.bn(self.conv(features)))           # (M, out_channel, P)
        # Max pooling over points in each pillar
        pooled   = torch.max(features, dim=2)[0]                   # (M, out_channel)

        # --- Scatter features back to pseudo-image ---
        # Extract correct batch indices
        batch_ids = coors[:, 0].long() # First column is batch index
        B = int(batch_ids.max().item()) + 1

        canvases = []
        for b in range(B):
            sel  = (batch_ids == b)
            if not torch.any(sel): # Skip if no voxels for this batch item
                 # Append an empty canvas of the correct shape and type
                 canvas = torch.zeros(
                     self.conv.out_channels,
                     self.y_l, self.x_l, # y_l first (height), x_l second (width)
                     device=device,
                     dtype=pooled.dtype # Match dtype (e.g., float16 under AMP)
                 )
                 canvases.append(canvas)
                 continue

            coors_batch = coors[sel] # Shape (Mb, 4) -> [b, z, y, x]
            feat_batch = pooled[sel] # Shape (Mb, out_channel)

            # Extract Y and X indices (check bounds)
            y_indices = coors_batch[:, 2].long() # Y index
            x_indices = coors_batch[:, 3].long() # X index

            # Create canvas for this batch item
            canvas = torch.zeros(
                self.conv.out_channels,
                self.y_l, self.x_l, # y_l first (height), x_l second (width)
                device=device,
                dtype=feat_batch.dtype # Match dtype
            )

            # Filter out-of-bounds indices (shouldn't happen with correct Voxelization)
            valid_mask = (x_indices >= 0) & (x_indices < self.x_l) & \
                         (y_indices >= 0) & (y_indices < self.y_l)

            if valid_mask.any():
                 # Scatter features using correct Y and X indices
                 canvas[:, y_indices[valid_mask], x_indices[valid_mask]] = feat_batch[valid_mask].t()

            canvases.append(canvas)

        # Stack canvases along batch dimension
        return torch.stack(canvases, dim=0) # (B, out_channel, Y_l, X_l)


class Backbone(nn.Module):
    # --- Keep your checkpointing implementation here ---
    def __init__(self, in_channel, out_channels, layer_nums, strides=[2,2,2]):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_channel
        for oc, ln, st in zip(out_channels, layer_nums, strides):
            seq = [
                nn.Conv2d(ch, oc, 3, stride=st, padding=1, bias=False),
                nn.BatchNorm2d(oc, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ]
            for _ in range(ln):
                seq += [
                    nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                    nn.BatchNorm2d(oc, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)
                ]
            self.blocks.append(nn.Sequential(*seq))
            ch = oc

    def forward(self, x):
        outs = []
        # use_reentrant=False is slightly faster / uses less Python overhead
        use_reentrant_strategy = False # Set based on PyTorch version/preference
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=use_reentrant_strategy)
            outs.append(x)
        return outs


# --- Neck and Head remain unchanged ---
class Neck(nn.Module):
    def __init__(self, in_channels, up_strides, out_channels):
        super().__init__()
        self.decoders = nn.ModuleList()
        for ic, us, oc in zip(in_channels, up_strides, out_channels):
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(ic, oc, us, stride=us, bias=False),
                nn.BatchNorm2d(oc, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ))

    def forward(self, feats):
        ups = [dec(f) for dec,f in zip(self.decoders, feats)]
        return torch.cat(ups, dim=1)

class Head(nn.Module):
    def __init__(self, in_chan, n_anchors, n_classes):
        super().__init__()
        # Calculate number of anchors based on input channel and n_classes
        # Assuming in_chan = neck_output_channels
        # head output channels = n_anchors * features_per_anchor
        self.n_anchors = n_anchors # Use the passed value

        self.conv_cls = nn.Conv2d(in_chan, self.n_anchors * n_classes, 1)
        self.conv_reg = nn.Conv2d(in_chan, self.n_anchors * 7,       1) # 7 = dx,dy,dz,dw,dl,dh,dyaw
        self.conv_dir = nn.Conv2d(in_chan, self.n_anchors * 2,       1) # 2 = direction classes
        
        # Initialize weights (standard practice)
        layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    if layer_id == 0: # Initialize bias for classification layer
                        b = -np.log((1-0.01)/0.01) # Focal loss prior init
                        nn.init.constant_(m.bias, b)
                    else:
                        nn.init.constant_(m.bias, 0)
                layer_id += 1 # Increment only for Conv layers considered

    def forward(self, x):
        return self.conv_cls(x), self.conv_reg(x), self.conv_dir(x)


class PointPillars(nn.Module):
    def __init__(self,
                 nclasses=3,
                 # Match voxel size used in SoftPillarLayer
                 voxel_size=(0.16,0.16,4),
                 pc_range=(0,-39.68,-3,69.12,39.68,1),
                 max_points=35,
                 # Match max_voxels used in SoftPillarLayer
                 max_voxels=(16000, 40000)):
        super().__init__()
        self.nclasses = nclasses

        # Use the modified layers
        self.pillar_layer   = SoftPillarLayer(voxel_size, pc_range, max_points, max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size, pc_range, in_channel=9, out_channel=64) # Input features = 9
        # Use the potentially reduced backbone/neck channels
        self.backbone       = Backbone(64, [64,128,256], [3,5,5])
        self.neck           = Neck([64,128,256], [1,2,4], [128,128,128])
        # Head input channel must match Neck output channel sum
        neck_out_channels = 384 # 64+64+64
        # Calculate n_anchors based on anchor generator setup below
        # Typically len(sizes) * len(rots) per class
        num_anchor_sizes = 3
        num_anchor_rots = 2
        n_anchors_per_loc_per_class = num_anchor_sizes * num_anchor_rots
        # Total anchors per location across all classes (used by head)
        self.n_anchors = n_anchors_per_loc_per_class * nclasses # 6 * 3 = 18? No, head expects per class usually. Check usage.
        # Head's n_anchors usually means anchors per location *per class* if separated by class later,
        # OR total anchors per location if classes are handled in the channel dimension.
        # Looking at Head output: n_anchors * n_classes for cls, confirms n_anchors means total anchors per loc.
        # Let's keep n_anchors as total per location.
        self.n_anchors_total_per_loc = n_anchors_per_loc_per_class # 6

        self.head = Head(neck_out_channels, n_anchors=self.n_anchors_total_per_loc, n_classes=nclasses)


        # --- Anchor Generator Setup ---
        # Define ranges, sizes, rotations based on KITTI common practice
        # Ensure these match the dataset and evaluation protocol
        # Example ranges per class (Car, Ped, Cyc) - adjust if needed
        ranges = [ # Y min/max different for Ped/Cyc? Check common configs.
            [0, -39.68, -0.6, 69.12, 39.68, -0.6],   # Car z-center approx -0.6 + 1.56/2 = 0.18
            [0, -39.68, -0.6, 69.12, 39.68, -0.6],   # Ped z-center approx -0.6 + 1.73/2 = 0.265
            [0, -39.68, -1.78, 69.12, 39.68, -1.78] # Cyc z-center approx -1.78 + 1.73/2 = -0.915
        ]
        # Example sizes per class [w, l, h] (check KITTI stats, common configs)
        sizes  = [
            [1.6, 3.9, 1.56], # Car size
            [0.6, 0.8, 1.73], # Pedestrian size
            [0.6, 1.76, 1.73] # Cyclist size
        ]
        rots   = [0, np.pi/2] # 0 and 90 degrees commonly used

        self.anchors_generator = Anchors(ranges=ranges, sizes=sizes, rotations=rots)
        # Note: The Anchors class needs to handle generating anchors for the specific feature map size

        # --- Target Assigner Setup ---
        # Common assignment thresholds for KITTI
        self.assigners  = [
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45}, # Car
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35}, # Ped
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35}, # Cyc
        ]
        # Ensure assigners list length matches nclasses if per-class assignment needed by anchor_target

        # --- Post-processing Params ---
        self.nms_pre    = 100 # Max boxes before NMS (per class) - reduce if memory issues persist?
        self.nms_thr    = 0.01 # Very low NMS threshold typical for BEV
        self.score_thr  = 0.1 # Min score to consider box before NMS
        self.max_num    = 50 # Max boxes to keep after NMS

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        # Input: List of Tensors [(N1, C), (N2, C), ...]
        batch_size = len(batched_pts)

        pillars, coors, nums = self.pillar_layer(batched_pts)
        # pillars: (M_total, P, C_pillar), coors: (M_total, 4) [b, z, y, x], nums: (M_total,)

        # Check for empty output from pillar_layer
        if pillars.numel() == 0:
             # Handle empty case: return empty predictions or structure expected by loss/eval
             print("Warning: Pillar layer produced empty output.")
             # Need to define expected output structure for empty case
             # Example for training:
             if mode == 'train':
                  # Return dummy tensors matching expected structure but size 0?
                  # Or handle in loss function? For now, let it proceed, check downstream.
                  # It might be better to return structure expected by anchor_target with 0 anchors.
                  pass # Let subsequent layers handle potentially empty tensors for now

        canvas  = self.pillar_encoder(pillars, coors, nums)
        # canvas: (B, C_enc, H_enc, W_enc) e.g., (B, 64, 248, 216)

        feats   = self.backbone(canvas)
        # feats: List [(B, C1, H1, W1), (B, C2, H2, W2), ...]

        fused   = self.neck(feats)
        # fused: (B, C_fused, H_out, W_out) e.g., (B, 192, 124, 108)

        cls_p, reg_p, dir_p = self.head(fused)
        # cls_p: (B, n_anchors*n_classes, H_out, W_out) e.g., (B, 6*3=18, 124, 108)
        # reg_p: (B, n_anchors*7, H_out, W_out)       e.g., (B, 6*7=42, 124, 108)
        # dir_p: (B, n_anchors*2, H_out, W_out)       e.g., (B, 6*2=12, 124, 108)

        # --- Mode handling ---
        if mode == 'train':
            # Generate anchors for the output feature map size
            fmap_h, fmap_w = cls_p.shape[-2:]
            device = cls_p.device
            feature_map_size = torch.tensor([fmap_h, fmap_w], device=device)

            # Ensure anchor generator is called correctly
            # anchors should be shape (H_out, W_out, num_anchor_types, 7) or similar
            # where num_anchor_types might be len(sizes)*len(rots)
            anchors = self.anchors_generator.get_multi_anchors(feature_map_size) # Get single set of anchors
            # Repeat anchors for each batch item
            batched_anchors = [anchors.clone() for _ in range(batch_size)] # Clone to avoid modifying base anchors

            # Assign targets
            # Ensure anchor_target function handles the batched anchors and GTs correctly
            tgt = anchor_target(batched_anchors=batched_anchors, # List of anchor tensors
                                batched_gt_bboxes=batched_gt_bboxes, # List of GT bbox tensors
                                batched_gt_labels=batched_gt_labels, # List of GT label tensors
                                assigners=self.assigners,
                                nclasses=self.nclasses)
            # tgt should be a dict containing batched tensors like 'batched_labels', 'batched_bbox_reg', etc.
            # Shapes should match B * total_num_anchors, e.g., (B, H*W*num_anchors_per_loc)

            return cls_p, reg_p, dir_p, tgt

        elif mode == 'val' or mode == 'test':
            # Generate anchors (same as train)
            fmap_h, fmap_w = cls_p.shape[-2:]
            device = cls_p.device
            feature_map_size = torch.tensor([fmap_h, fmap_w], device=device)
            anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
            batched_anchors = [anchors.clone() for _ in range(batch_size)]

            # Decode predictions for each batch item
            outs = []
            for i in range(batch_size):
                 # Pass single batch item preds and corresponding anchors
                 decoded = self._decode_single(cls_p[i], reg_p[i], dir_p[i], batched_anchors[i])
                 outs.append(decoded)
            return outs # List of dictionaries per batch item

        else:
             raise ValueError(f"Unknown mode: {mode}")


    def _decode_single(self, c_p, r_p, d_p, anchors):
        # c_p: (n_anchors*n_classes, H, W)
        # r_p: (n_anchors*7, H, W)
        # d_p: (n_anchors*2, H, W)
        # anchors: (H, W, n_anchor_types, 7) or similar flattened shape

        # --- Reshape predictions and anchors ---
        # Assuming n_anchors_total_per_loc = 6
        num_anchors = self.n_anchors_total_per_loc
        bbox_cls = c_p.permute(1, 2, 0).reshape(-1, self.nclasses) # (H*W*n_anchors, n_classes)
        bbox_reg = r_p.permute(1, 2, 0).reshape(-1, 7)           # (H*W*n_anchors, 7)
        dir_cls  = d_p.permute(1, 2, 0).reshape(-1, 2)           # (H*W*n_anchors, 2)
        
        # Ensure anchors are flattened to match predictions
        anchors_flat = anchors.reshape(-1, 7)                      # (H*W*n_anchors, 7)
        
        # --- Process predictions ---
        bbox_cls = bbox_cls.sigmoid() # Apply sigmoid to classification scores
        dir_cls  = torch.argmax(dir_cls, dim=1) # Get direction class (0 or 1)

        # --- Top-K pre-NMS selection ---
        # Get top scores across all classes for each anchor
        top_scores, _ = bbox_cls.max(dim=1)
        # Select top k anchors based on highest score across classes
        if top_scores.numel() == 0: # Handle empty predictions
             return {'lidar_bboxes': np.array([]), 'labels': np.array([]), 'scores': np.array([])}
             
        nms_pre_count = min(self.nms_pre, top_scores.numel()) # Adjust if fewer anchors than nms_pre
        top_scores, top_inds = top_scores.topk(nms_pre_count)

        # Filter all tensors based on top_inds
        bbox_cls = bbox_cls[top_inds]
        bbox_reg = bbox_reg[top_inds]
        dir_cls  = dir_cls[top_inds]
        anchors_flat = anchors_flat[top_inds]

        # --- Decode BBoxes ---
        # anchors2bboxes should take anchors and regression predictions
        bbox3d = anchors2bboxes(anchors_flat, bbox_reg) # (nms_pre_count, 7) [x, y, z, w, l, h, yaw]

        # --- NMS per class ---
        # Create 2D BEV boxes for NMS [x1, y1, x2, y2, score] or [x,y,w,l,yaw,score]?
        # nms_cuda likely expects [x1, y1, x2, y2, score]
        # Convert [x, y, w, l] to [x1, y1, x2, y2]
        xy_center = bbox3d[:, [0, 1]]
        wl = bbox3d[:, [3, 4]] # Assuming order w, l
        bev_boxes_xyxy = torch.cat([xy_center - wl / 2, xy_center + wl / 2], dim=1) # (nms_pre_count, 4)

        final_bboxes, final_labels, final_scores = [], [], []
        for cls_id in range(self.nclasses):
            # Get scores for the current class
            class_scores = bbox_cls[:, cls_id]
            # Filter by score threshold
            score_mask = class_scores > self.score_thr
            if not score_mask.any():
                continue

            # Select boxes, scores, directions for this class above threshold
            scores_C = class_scores[score_mask]
            boxes_bev_C = bev_boxes_xyxy[score_mask]
            boxes_3d_C = bbox3d[score_mask]
            dir_C = dir_cls[score_mask]

            # Perform NMS
            # Ensure nms_cuda input format is correct
            keep_indices = nms_cuda(boxes=boxes_bev_C, scores=scores_C, thresh=self.nms_thr)
            # keep_indices = torchvision.ops.nms(boxes_bev_C, scores_C, iou_threshold=self.nms_thr) # Alternative if nms_cuda unavailable

            if keep_indices.numel() == 0:
                continue

            # Select kept boxes and apply direction correction
            final_bboxes_C = boxes_3d_C[keep_indices]
            final_scores_C = scores_C[keep_indices]
            final_dir_C = dir_C[keep_indices]

            # Correct yaw based on direction prediction
            # Yaw is likely the last element (index 6)
            yaw = final_bboxes_C[:, 6]
            # Apply correction based on dir_cls (0 or 1)
            # Assumes dir_cls=0 means yaw is correct, dir_cls=1 means yaw + pi
            dir_correction = (1 - final_dir_C.float()) * torch.pi # Correction is pi if dir_cls=1 (assuming 1 means opposite dir)
            corrected_yaw = yaw + dir_correction
            # Limit period to [-pi, pi) - CHECK limit_period definition
            # limit_period might expect offset=0.5 for [-pi,pi), offset=1 for [0, 2pi)?
            final_bboxes_C[:, 6] = limit_period(corrected_yaw, offset=0.5, period=torch.pi * 2) # Adjust offset/period if needed

            final_bboxes.append(final_bboxes_C)
            final_labels.append(torch.full((len(keep_indices),), cls_id, dtype=torch.long, device=final_scores_C.device))
            final_scores.append(final_scores_C)

        # Combine results from all classes
        if not final_bboxes:
             return {'lidar_bboxes': np.array([]), 'labels': np.array([]), 'scores': np.array([])}

        final_bboxes = torch.cat(final_bboxes, dim=0)
        final_labels = torch.cat(final_labels, dim=0)
        final_scores = torch.cat(final_scores, dim=0)

        # Limit to max_num boxes overall
        if final_bboxes.shape[0] > self.max_num:
            topk_scores, topk_indices = final_scores.topk(self.max_num)
            final_bboxes = final_bboxes[topk_indices]
            final_labels = final_labels[topk_indices]
            final_scores = topk_scores # Already have the top scores

        return {
            'lidar_bboxes': final_bboxes.detach().cpu().numpy(),
            'labels':       final_labels.detach().cpu().numpy(),
            'scores':       final_scores.detach().cpu().numpy()
        }