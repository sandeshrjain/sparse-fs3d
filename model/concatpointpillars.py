import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.anchors import Anchors, anchor_target, anchors2bboxes
from ops import Voxelization, nms_cuda, AdaptiveVoxelizationGPU
from utils import limit_period




class PillarLayer(nn.Module):
    def __init__(self, max_voxels_dense=12000, max_voxels_sparse=6000):
        super(PillarLayer, self).__init__()
        self.voxel_layer = AdaptiveVoxelizationGPU(max_voxels_dense=max_voxels_dense, max_voxels_sparse=max_voxels_sparse)

    @torch.no_grad()
    def forward(self, batched_pts):
        """
        Args:
            batched_pts: list[tensor], len(batched_pts) = batch_size
        Returns:
            dict with "dense" and "sparse" outputs containing:
                - voxels: (N, max_points_per_voxel, C)
                - coors_batch: (N, 1 + 3) with batch indices prepended to coordinates
                - num_points_per_voxel: (N,)
        """
        dense_voxels, dense_coors, dense_num_points = [], [], []
        sparse_voxels, sparse_coors, sparse_num_points = [], [], []

        for i, pts in enumerate(batched_pts):
            # Apply adaptive voxelization
            output = self.voxel_layer(pts)
            dense = output['dense']
            sparse = output['sparse']

            # Add batch indices to voxel coordinates
            dense_coors.append(F.pad(dense['coors'], (1, 0), value=i))
            sparse_coors.append(F.pad(sparse['coors'], (1, 0), value=i))

            # Collect dense and sparse results
            dense_voxels.append(dense['voxels'])
            sparse_voxels.append(sparse['voxels'])
            dense_num_points.append(dense['num_points'])
            sparse_num_points.append(sparse['num_points'])

        # Combine all results across the batch
        dense_result = {
            "voxels": torch.cat(dense_voxels, dim=0),                # (total_dense_voxels, max_points, C)
            "coors_batch": torch.cat(dense_coors, dim=0),            # (total_dense_voxels, 1 + 3)
            "num_points": torch.cat(dense_num_points, dim=0)         # (total_dense_voxels,)
        }
        sparse_result = {
            "voxels": torch.cat(sparse_voxels, dim=0),               # (total_sparse_voxels, max_points, C)
            "coors_batch": torch.cat(sparse_coors, dim=0),           # (total_sparse_voxels, 1 + 3)
            "num_points": torch.cat(sparse_num_points, dim=0)        # (total_sparse_voxels,)
        }

        return {"dense": dense_result, "sparse": sparse_result}



class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])+1
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])+1

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device

        # Prevent zero division in npoints_per_pillar
        npoints_per_pillar_safe = npoints_per_pillar.clamp(min=1)

        # Offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar_safe[:, None, None]

        # Offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset)

        # Encode features
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)

        # Mask invalid points
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)
        mask = voxel_ids[:, None] < npoints_per_pillar_safe[None, :]
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]

        # Embedding
        features = features.permute(0, 2, 1).contiguous()
        features = F.relu(self.bn(self.conv(features)))
        pooling_features = torch.max(features, dim=-1)[0]

        # Scatter to canvas
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            # Fix: Cast cur_coors[:, 1:3] to long
            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            # print("max coors x:", cur_coors[:, 1].max().item(), " vs self.x_l:", self.x_l)
            # print("max coors y:", cur_coors[:, 2].max().item(), " vs self.y_l:", self.y_l)
            # print("min coors x:", cur_coors[:, 1].min().item())
            # print("min coors y:", cur_coors[:, 2].min().item())

            canvas[cur_coors[:, 1].long(), cur_coors[:, 2].long()] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)

        batched_canvas = torch.stack(batched_canvas, dim=0)
        return batched_canvas



class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


# class Neck(nn.Module):
#     def __init__(self, in_channels, upsample_strides, out_channels):
#         super().__init__()
#         assert len(in_channels) == len(upsample_strides)
#         assert len(upsample_strides) == len(out_channels)

#         self.decoder_blocks = nn.ModuleList()
#         for i in range(len(in_channels)):
#             decoder_block = []
#             decoder_block.append(nn.ConvTranspose2d(in_channels[i], 
#                                                     out_channels[i], 
#                                                     upsample_strides[i], 
#                                                     stride=upsample_strides[i],
#                                                     bias=False))
#             decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
#             decoder_block.append(nn.ReLU(inplace=True))

#             self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
#         # in consitent with mmdet3d
#         for m in self.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # def forward(self, x):
    #     '''
    #     x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
    #     return: (bs, 384, 248, 216)
    #     '''
    #     outs = []
    #     for i in range(len(self.decoder_blocks)):
    #         xi = self.decoder_blocks[i](x[i]) # (bs, 128, 248, 216)
    #         outs.append(xi)
    #     out = torch.cat(outs, dim=1)
    #     return out

class Neck(nn.Module):
    def __init__(self, in_channels_dense, in_channels_sparse, upsample_strides, out_channels, fusion_method="concat"):
        super(Neck, self).__init__()
        assert len(in_channels_dense) == len(upsample_strides)
        assert len(in_channels_sparse) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.fusion_method = fusion_method

        # Decoder for dense features
        self.decoder_dense = nn.ModuleList()
        for i in range(len(in_channels_dense)):
            decoder_block = [
                nn.ConvTranspose2d(in_channels_dense[i], out_channels[i], upsample_strides[i],
                                   stride=upsample_strides[i], bias=False),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ]
            self.decoder_dense.append(nn.Sequential(*decoder_block))

        # Decoder for sparse features
        self.decoder_sparse = nn.ModuleList()
        for i in range(len(in_channels_sparse)):
            decoder_block = [
                nn.ConvTranspose2d(in_channels_sparse[i], out_channels[i], upsample_strides[i],
                                   stride=upsample_strides[i], bias=False),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ]
            self.decoder_sparse.append(nn.Sequential(*decoder_block))

        # Downsampling after fusion
        self.downsampling_layers = nn.ModuleList()
        for i in range(len(out_channels)):
            self.downsampling_layers.append(nn.Sequential(
                nn.Conv2d(out_channels[i] * 2, out_channels[i], kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ))

        # Initialization (consistent with mmdet3d)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x_dense, x_sparse):
        outs_dense = []
        outs_sparse = []

        for i in range(len(self.decoder_dense)):
            dense_feature = self.decoder_dense[i](x_dense[i])
            sparse_feature = self.decoder_sparse[i](x_sparse[i])

            # Align spatial dimensions at each scale
            target_shape = (min(dense_feature.shape[2], sparse_feature.shape[2]),
                            min(dense_feature.shape[3], sparse_feature.shape[3]))

            dense_feature = F.interpolate(dense_feature, size=target_shape, mode="bilinear", align_corners=False)
            sparse_feature = F.interpolate(sparse_feature, size=target_shape, mode="bilinear", align_corners=False)

            # Concatenate dense and sparse features
            fused_output = torch.cat([dense_feature, sparse_feature], dim=1)

            # Downsample to match `out_channels`
            fused_output = self.downsampling_layers[i](fused_output)
            outs_dense.append(fused_output)

        # Ensure consistent spatial dimensions across all scales
        target_height = min([feat.shape[2] for feat in outs_dense])
        target_width = min([feat.shape[3] for feat in outs_dense])
        target_shape = (target_height, target_width)

        # Resize all fused features to the target shape
        fused_features = [F.interpolate(feat, size=target_shape, mode="bilinear", align_corners=False) for feat in outs_dense]

        # Concatenate aligned fused features across scales
        final_output = torch.cat(fused_features, dim=1)
        return final_output











class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class PointPillars(nn.Module):
    def __init__(self,
                 nclasses=3,
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=32,
                 max_voxels=(16000, 40000)):
        super().__init__()
        self.nclasses = nclasses

        # Pillar Layer with Adaptive Voxelization
        self.pillar_layer = PillarLayer(max_voxels_dense=max_voxels[0], max_voxels_sparse=max_voxels[1])

        # Encoders for dense and sparse streams
        self.pillar_encoder_dense = PillarEncoder(
            voxel_size=self.pillar_layer.voxel_layer.voxel_size_dense,  # Reference learnable voxel size
            point_cloud_range=point_cloud_range,
            in_channel=9,
            out_channel=64
        )
        self.pillar_encoder_sparse = PillarEncoder(
            voxel_size=self.pillar_layer.voxel_layer.voxel_size_sparse,  # Reference learnable voxel size
            point_cloud_range=point_cloud_range,
            in_channel=9,
            out_channel=64
        )

        # Backbones
        self.backbone_dense = Backbone(in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5])
        self.backbone_sparse = Backbone(in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5])

        # Neck for fusion
        self.neck = Neck(
            in_channels_dense=[64, 128, 256],
            in_channels_sparse=[64, 128, 256],
            upsample_strides=[1, 2, 4],
            out_channels=[128, 128, 128],
            fusion_method="concat"
        )


        # Detection Head
        self.head = Head(in_channel=384, n_anchors=2 * nclasses, n_classes=nclasses)

        # Anchors
        ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                  [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                  [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations = [0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges, sizes=sizes, rotations=rotations)

        # Training configurations
        self.assigners = [{'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
                          {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
                          {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45}]

        # Validation and Testing configurations
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return [], [], []
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result


    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        """
        Args:
            batched_pts: List of input point clouds
            mode: 'train', 'val', or 'test'
            batched_gt_bboxes: Ground truth bounding boxes (for training)
            batched_gt_labels: Ground truth labels (for training)
        Returns:
            Detection results or training targets
        """
        batch_size = len(batched_pts)

        # Step 1: Adaptive Voxelization (outputs dense and sparse streams)
        outputs = self.pillar_layer(batched_pts)
        dense_output, sparse_output = outputs['dense'], outputs['sparse']

        # Step 2: Pillar Encoding
        dense_features = self.pillar_encoder_dense(
            dense_output["voxels"], dense_output["coors_batch"], dense_output["num_points"])
        sparse_features = self.pillar_encoder_sparse(
            sparse_output["voxels"], sparse_output["coors_batch"], sparse_output["num_points"])

        # Step 3: Backbone Processing for Dense Stream
        dense_backbone_features = self.backbone_dense(dense_features)
        # dense_backbone_features: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]

        # Step 4: Backbone Processing for Sparse Stream
        sparse_backbone_features = self.backbone_sparse(sparse_features)
        # sparse_backbone_features: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]

        # Step 5: Neck Fusion
        fused_features = self.neck(dense_backbone_features, sparse_backbone_features)

        # Step 6: Detection Head
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(fused_features)

        # Step 7: Anchor Generation
        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), dtype=torch.int32, device=device)
        # print(f"Feature map size used in anchors: {feature_map_size}")
        # print(f"bbox_cls_pred spatial size: {bbox_cls_pred.size()[-2:]}")

        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]


        # Step 8: Handle Modes
        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors,
                                               batched_gt_bboxes=batched_gt_bboxes,
                                               batched_gt_labels=batched_gt_labels,
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode in ['val', 'test']:
            results = self.get_predicted_bboxes(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors)
            return results
        else:
            raise ValueError("Unsupported mode. Choose from 'train', 'val', or 'test'.")
