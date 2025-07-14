import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.anchors import Anchors, anchor_target, anchors2bboxes
from ops import Voxelization, nms_cuda
from utils import limit_period


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
            weights = weights /  (weights.sum() + 1e-6)

            padded = torch.zeros((self.max_points_per_voxel, pts_top.size(1)), device=device)
            padded[:pts_top.size(0)] = pts_top * weights[:, None]

            voxel_features.append(padded)
            voxel_coords.append(coord)
            num_points_per_voxel.append(len(pts_top))

            if len(voxel_features) >= self.max_voxels:
                break

        if len(voxel_features) == 0:
            return torch.empty(0, self.max_points_per_voxel, points.size(1), device=device), \
                   torch.empty(0, 3, dtype=torch.int32, device=device), \
                   torch.empty(0, dtype=torch.int32, device=device)

        voxel_features = torch.stack(voxel_features, dim=0)
        voxel_coords = torch.stack(voxel_coords, dim=0)
        num_points_per_voxel = torch.tensor(num_points_per_voxel, device=device, dtype=torch.int32)

        return voxel_features, voxel_coords, num_points_per_voxel


class SoftPillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = GaussianSoftVoxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_num_points,
            max_voxels=max_voxels
        )

    @torch.no_grad()
    def forward(self, batched_pts):
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_out = self.voxel_layer(pts)
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_out)

        pillars = torch.cat(pillars, dim=0)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)
        coors_batch = [F.pad(c, (1, 0), value=i) for i, c in enumerate(coors)]
        coors_batch = torch.cat(coors_batch, dim=0)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

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
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp
        # In consitent with mmdet3d. 
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, in_channel, self.y_l, self.x_l)
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


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], 
                                                    out_channels[i], 
                                                    upsample_strides[i], 
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i]) # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


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
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=32,
                 max_voxels=20000):
        super().__init__()
        self.nclasses = nclasses

        self.pillar_layer = SoftPillarLayer(voxel_size=voxel_size,
                                            point_cloud_range=point_cloud_range,
                                            max_num_points=max_num_points,
                                            max_voxels=max_voxels)

        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size,
                                            point_cloud_range=point_cloud_range,
                                            in_channel=9,
                                            out_channel=64)

        self.backbone = Backbone(in_channel=64,
                                 out_channels=[64, 128, 256],
                                 layer_nums=[3, 5, 5])

        self.neck = Neck(in_channels=[64, 128, 256],
                         upsample_strides=[1, 2, 4],
                         out_channels=[128, 128, 128])

        self.head = Head(in_channel=384, n_anchors=2 * nclasses, n_classes=nclasses)

        self.anchors_generator = Anchors(
            ranges=[[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57])

        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        bs = len(batched_pts)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        feat_canvas = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        xs = self.backbone(feat_canvas)
        x = self.neck(xs)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)
        if torch.isnan(bbox_cls_pred).any() or torch.isnan(bbox_pred).any():
            print("[ERROR] NaN detected in bbox prediction!")
            print("bbox_cls_pred stats:", bbox_cls_pred.min().item(), bbox_cls_pred.max().item())
            print("bbox_pred stats:", bbox_pred.min().item(), bbox_pred.max().item())
            import pdb; pdb.set_trace()

        if torch.isinf(bbox_cls_pred).any() or torch.isinf(bbox_pred).any():
            print("[ERROR] Inf detected in bbox prediction!")
            import pdb; pdb.set_trace()

        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(bs)]

        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors,
                                               batched_gt_bboxes=batched_gt_bboxes,
                                               batched_gt_labels=batched_gt_labels,
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict

        elif mode in ['val', 'test']:
            return self.get_predicted_bboxes(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors)
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")

    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        bs = bbox_cls_pred.size(0)
        results = []
        for i in range(bs):
            result = self.get_predicted_bboxes_single(
                bbox_cls_pred[i], bbox_pred[i], bbox_dir_cls_pred[i], batched_anchors[i])
            results.append(result)
        return results

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)

        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.argmax(bbox_dir_cls_pred, dim=1)

        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            score_mask = bbox_cls_pred[:, i] > self.score_thr
            if score_mask.sum() == 0:
                continue
            scores = bbox_cls_pred[score_mask, i]
            boxes2d = bbox_pred2d[score_mask]
            boxes3d = bbox_pred[score_mask]
            dirs = bbox_dir_cls_pred[score_mask]

            keep = nms_cuda(boxes2d, scores, self.nms_thr)
            boxes3d = boxes3d[keep]
            scores = scores[keep]
            dirs = dirs[keep]

            boxes3d[:, -1] = limit_period(boxes3d[:, -1], 1, np.pi) + (1 - dirs) * np.pi

            ret_bboxes.append(boxes3d)
            ret_labels.append(torch.full((boxes3d.shape[0],), i, dtype=torch.long, device=boxes3d.device))
            ret_scores.append(scores)

        if not ret_bboxes:
            return {'lidar_bboxes': [], 'labels': [], 'scores': []}

        ret_bboxes = torch.cat(ret_bboxes)
        ret_labels = torch.cat(ret_labels)
        ret_scores = torch.cat(ret_scores)

        if ret_bboxes.shape[0] > self.max_num:
            topk = ret_scores.topk(self.max_num).indices
            ret_bboxes = ret_bboxes[topk]
            ret_labels = ret_labels[topk]
            ret_scores = ret_scores[topk]

        return {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }