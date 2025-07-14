import argparse
import numpy as np
import os
import torch
from tqdm import tqdm

from utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, write_pickle, write_label, \
    iou2d, iou3d_camera, iou_bev
from dataset import Kitti, get_dataloader
from model import PointPillars


def get_score_thresholds(tp_scores, total_num_valid_gt, num_sample_pts=41):
    score_thresholds = []
    tp_scores = sorted(tp_scores)[::-1]
    cur_recall, pts_ind = 0, 0
    for i, score in enumerate(tp_scores):
        lrecall = (i + 1) / total_num_valid_gt
        rrecall = (i + 2) / total_num_valid_gt

        if i == len(tp_scores) - 1:
            score_thresholds.append(score)
            break

        if (lrecall + rrecall) / 2 < cur_recall:
            continue

        score_thresholds.append(score)
        pts_ind += 1
        cur_recall = pts_ind / (num_sample_pts - 1)
    return score_thresholds


def do_eval(det_results, gt_results, CLASSES, saved_path):
    '''
    det_results: dict idx->detections
    gt_results: dict idx->ground truth infos
    CLASSES: dict class_name->index
    '''
    assert len(det_results) == len(gt_results)
    f = open(os.path.join(saved_path, 'eval_results.txt'), 'w')

    # 1. calculate IoUs
    ious = {'bbox_2d': [], 'bbox_bev': [], 'bbox_3d': []}
    ids = list(sorted(gt_results.keys()))

    for idx in ids:
        gt = gt_results[idx]['annos']
        dt = det_results[idx]

        # ----- 2D IoU -----
        gt_2d = gt['bbox'].astype(np.float32)
        dt_2d = dt['bbox'].astype(np.float32)
        if gt_2d.shape[0] == 0 or dt_2d.shape[0] == 0:
            iou2d_v = np.zeros((gt_2d.shape[0], dt_2d.shape[0]), dtype=np.float32)
        else:
            iou2d_v = iou2d(
                torch.from_numpy(gt_2d).cuda(),
                torch.from_numpy(dt_2d).cuda()
            ).cpu().numpy()
        ious['bbox_2d'].append(iou2d_v)

        # ----- BEV IoU -----
        gt_loc = gt['location'].astype(np.float32)
        gt_dim = gt['dimensions'].astype(np.float32)
        gt_ry  = gt['rotation_y'].astype(np.float32)
        dt_loc = dt['location'].astype(np.float32)
        dt_dim = dt['dimensions'].astype(np.float32)
        dt_ry  = dt['rotation_y'].astype(np.float32)

        if gt_loc.shape[0] == 0 or dt_loc.shape[0] == 0:
            iou_bev_v = np.zeros((gt_loc.shape[0], dt_loc.shape[0]), dtype=np.float32)
        else:
            # ensure rotation is column vector
            if gt_ry.ndim == 1:
                gt_ry = gt_ry.reshape(-1, 1)
            if dt_ry.ndim == 1:
                dt_ry = dt_ry.reshape(-1, 1)

            gt_bev = np.concatenate([gt_loc[:, [0, 2]], gt_dim[:, [0, 2]], gt_ry], axis=1)
            dt_bev = np.concatenate([dt_loc[:, [0, 2]], dt_dim[:, [0, 2]], dt_ry], axis=1)

            iou_bev_v = iou_bev(
                torch.from_numpy(gt_bev).cuda(),
                torch.from_numpy(dt_bev).cuda()
            ).cpu().numpy()
        ious['bbox_bev'].append(iou_bev_v)

        # ----- 3D IoU -----
        # ensure rotation is column vector before concatenation
        if gt_ry.ndim == 1:
            gt_ry = gt_ry.reshape(-1, 1)
        if dt_ry.ndim == 1:
            dt_ry = dt_ry.reshape(-1, 1)

        gt_3d = np.concatenate([gt_loc, gt_dim, gt_ry], axis=1)
        dt_3d = np.stack([dt_loc, dt_dim, dt_ry], axis=1)

        if gt_3d.shape[0] == 0 or dt_3d.shape[0] == 0:
            iou3d_v = np.zeros((gt_3d.shape[0], dt_3d.shape[0]), dtype=np.float32)
        else:
            iou3d_v = iou3d_camera(
                torch.from_numpy(gt_3d).cuda(),
                torch.from_numpy(dt_3d).cuda()
            ).cpu().numpy()
        ious['bbox_3d'].append(iou3d_v)

    # thresholds & heights
    MIN_IOUS = {
        'Pedestrian': [0.1, 0.5, 0.5],
        'Cyclist':    [0.1, 0.5, 0.5],
        'Car':        [0.1, 0.7, 0.7]
    }
    MIN_HEIGHT = [40, 25, 25]

    overall_results = {}
    # For each IoU type compute per-class AP/AOS
    for e_ind, eval_type in enumerate(['bbox_2d', 'bbox_bev', 'bbox_3d']):
        eval_ious = ious[eval_type]
        eval_ap_results, eval_aos_results = {}, {}
        for cls in CLASSES:
            eval_ap_results[cls]  = []
            eval_aos_results[cls] = []
            thr = MIN_IOUS[cls][e_ind]

            for diff in [0, 1, 2]:
                # collect TP scores and ignore flags
                total_gt_ignores, total_det_ignores = [], []
                total_dc_bboxes, total_scores = [], []
                total_gt_alpha, total_det_alpha = [], []

                for i, idx in enumerate(ids):
                    gt = gt_results[idx]['annos']
                    dt = det_results[idx]
                    iou_mat = eval_ious[i]

                    # ground truth ignores
                    gt_ign, dc = [], []
                    for j, name in enumerate(gt['name']):
                        ign = (gt['difficulty'][j] < 0) or (gt['difficulty'][j] > diff)
                        if name == cls:
                            vc = 1
                        elif cls == 'Pedestrian' and name == 'Person_sitting':
                            vc = 0
                        elif cls == 'Car' and name == 'Van':
                            vc = 0
                        else:
                            vc = -1

                        if vc == 1 and not ign:
                            gt_ign.append(0)
                        elif vc == 0 or (vc == 1 and ign):
                            gt_ign.append(1)
                        else:
                            gt_ign.append(-1)
                        if name == 'DontCare':
                            dc.append(gt['bbox'][j])
                    total_gt_ignores.append(gt_ign)
                    total_dc_bboxes.append(np.array(dc))
                    total_gt_alpha.append(gt['alpha'])

                    # detection ignores & scores
                    heights = dt['bbox'][:, 3] - dt['bbox'][:, 1]
                    det_ign = []
                    for j, name in enumerate(dt['name']):
                        if heights[j] < MIN_HEIGHT[diff]:
                            det_ign.append(1)
                        elif name == cls:
                            det_ign.append(0)
                        else:
                            det_ign.append(-1)
                    total_det_ignores.append(det_ign)
                    total_scores.append(dt['score'])
                    total_det_alpha.append(dt['alpha'])

                # TP-score thresholds for PR curve
                tp_scores = []
                for i, idx in enumerate(ids):
                    iou_mat = eval_ious[i]
                    gt_ign  = total_gt_ignores[i]
                    det_ign = total_det_ignores[i]
                    scores  = total_scores[i]
                    nn, mm = iou_mat.shape
                    assigned = np.zeros((mm,), dtype=bool)

                    for j in range(nn):
                        if gt_ign[j] == -1:
                            continue
                        best_k, best_sc = -1, -1
                        for k in range(mm):
                            if (not assigned[k] and det_ign[k] >= 0
                                and iou_mat[j, k] > thr
                                and scores[k] > best_sc):
                                best_k, best_sc = k, scores[k]
                        if best_k >= 0:
                            assigned[best_k] = True
                            if det_ign[best_k] == 0 and gt_ign[j] == 0:
                                tp_scores.append(best_sc)

                total_valid_gt = sum(np.sum(np.array(g) == 0) for g in total_gt_ignores)
                score_thresholds = get_score_thresholds(tp_scores, total_valid_gt)

                # compute PR curve, mAP and (if 2D) mAOS
                tps, fns, fps, aos_list = [], [], [], []
                for s_thr in score_thresholds:
                    tp = fn = fp = aos = 0
                    for i, idx in enumerate(ids):
                        iou_mat = eval_ious[i]
                        gt_ign  = total_gt_ignores[i]
                        det_ign = total_det_ignores[i]
                        gt_alpha = total_gt_alpha[i]
                        dt_alpha = total_det_alpha[i]
                        scores   = total_scores[i]
                        nn, mm = iou_mat.shape
                        assigned = np.zeros((mm,), dtype=bool)

                        for j in range(nn):
                            if gt_ign[j] == -1:
                                continue
                            match_id, match_iou = -1, -1
                            for k in range(mm):
                                if (not assigned[k] and det_ign[k] >= 0
                                    and scores[k] >= s_thr
                                    and iou_mat[j, k] > thr):
                                    if det_ign[k] == 0 and iou_mat[j, k] > match_iou:
                                        match_iou, match_id = iou_mat[j, k], k
                                    elif det_ign[k] == 1 and match_iou < 0:
                                        match_id = k
                            if match_id >= 0:
                                assigned[match_id] = True
                                if det_ign[match_id] == 0 and gt_ign[j] == 0:
                                    tp += 1
                                    if eval_type == 'bbox_2d':
                                        aos += (1 + np.cos(gt_alpha[j] - dt_alpha[match_id])) / 2
                            else:
                                if gt_ign[j] == 0:
                                    fn += 1

                        for k in range(mm):
                            if det_ign[k] == 0 and scores[k] >= s_thr and not assigned[k]:
                                fp += 1

                        # handle DontCare for 2D
                        if eval_type == 'bbox_2d':
                            dc = total_dc_bboxes[i]
                            dt_b = det_results[idx]['bbox']
                            if len(dc) > 0 and dt_b.size > 0:
                                iou_dc = iou2d(
                                    torch.from_numpy(dt_b).cuda(),
                                    torch.from_numpy(dc).cuda(),
                                    metric=1
                                ).cpu().numpy().T
                                for j in range(len(dc)):
                                    for k in range(len(dt_b)):
                                        if (total_det_ignores[i][k] == 0
                                            and scores[k] >= s_thr
                                            and not assigned[k]
                                            and iou_dc[j, k] > thr):
                                            fp -= 1
                                            assigned[k] = True

                    tps.append(tp)
                    fns.append(fn)
                    fps.append(fp)
                    if eval_type == 'bbox_2d':
                        aos_list.append(aos)

                tps, fns, fps = np.array(tps), np.array(fns), np.array(fps)
                precisions = tps / (tps + fps + 1e-8)
                for i in range(len(precisions)):
                    precisions[i] = precisions[i:].max()

                mAP = sum(precisions[0::4]) / 11 * 100
                eval_ap_results[cls].append(mAP)

                if eval_type == 'bbox_2d':
                    aos_arr = np.array(aos_list)
                    similarity = aos_arr / (tps + fps + 1e-8)
                    for i in range(len(similarity)):
                        similarity[i] = similarity[i:].max()
                    mAOS = sum(similarity[0::4]) / 11 * 100
                    eval_aos_results[cls].append(mAOS)

        # print and log
        print(f'=========={eval_type.upper()}==========')
        print(f'=========={eval_type.upper()}==========', file=f)
        for k, v in eval_ap_results.items():
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)

        if eval_type == 'bbox_2d':
            print(f'==========AOS==========')
            print(f'==========AOS==========', file=f)
            for k, v in eval_aos_results.items():
                print(f'{k} AOS@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
                print(f'{k} AOS@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)

        overall_results[eval_type] = np.mean(list(eval_ap_results.values()), 0)
        if eval_type == 'bbox_2d':
            overall_results['AOS'] = np.mean(list(eval_aos_results.values()), 0)

    # final overall
    print(f'\n==========Overall==========')
    print(f'\n==========Overall==========', file=f)
    for k, v in overall_results.items():
        print(f'{k} AP: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
        print(f'{k} AP: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)

    f.close()


def main(args):
    val_dataset = Kitti(data_root=args.data_root, split='val')
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    CLASSES = Kitti.CLASSES
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}

    if not args.no_cuda:
        model = PointPillars(nclasses=args.nclasses).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=args.nclasses)
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu'))
        )

    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    saved_submit = os.path.join(saved_path, 'submit')
    os.makedirs(saved_submit, exist_ok=True)

    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        format_results = {}
        print('Predicting and Formatting the results.')
        for i, data in enumerate(tqdm(val_dataloader)):
            if not args.no_cuda:
                # move tensors
                for k in data:
                    for j, x in enumerate(data[k]):
                        if torch.is_tensor(x):
                            data[k][j] = x.cuda()

            batch_pts     = data['batched_pts']
            batch_gts     = data['batched_gt_bboxes']
            batch_labels  = data['batched_labels']
            batch_results = model(
                batched_pts=batch_pts,
                mode='val',
                batched_gt_bboxes=batch_gts,
                batched_gt_labels=batch_labels
            )

            for j, result in enumerate(batch_results):
                fr = {
                    'name': [], 'truncated': [], 'occluded': [], 'alpha': [],
                    'bbox': [], 'dimensions': [], 'location': [],
                    'rotation_y': [], 'score': []
                }
                if isinstance(result, tuple):
                    result = {
                        'lidar_bboxes': result[0],
                        'labels': result[1],
                        'scores': result[2]
                    }

                cal = data['batched_calib_info'][j]
                tr, r0, P2 = (
                    cal['Tr_velo_to_cam'].astype(np.float32),
                    cal['R0_rect'].astype(np.float32),
                    cal['P2'].astype(np.float32)
                )
                img_shp = data['batched_img_info'][j]['image_shape']
                idx     = data['batched_img_info'][j]['image_idx']

                if len(result['lidar_bboxes']) == 0:
                    format_results[idx] = {
                        'name':       np.array([]),
                        'truncated':  np.array([]),
                        'occluded':   np.array([]),
                        'alpha':      np.array([]),
                        'bbox':       np.empty((0, 4), dtype=np.float32),
                        'dimensions': np.empty((0, 3), dtype=np.float32),
                        'location':   np.empty((0, 3), dtype=np.float32),
                        'rotation_y': np.empty((0, 1), dtype=np.float32),
                        'score':      np.array([])
                    }
                    continue

                rf = keep_bbox_from_image_range(result, tr, r0, P2, img_shp)
                rf = keep_bbox_from_lidar_range(rf, pcd_limit_range)

                for (lb, lbl, sc, bb2d, cb) in zip(
                    rf['lidar_bboxes'], rf['labels'],
                    rf['scores'],       rf['bboxes2d'],
                    rf['camera_bboxes']
                ):
                    fr['name'].append(LABEL2CLASSES[lbl])
                    fr['truncated'].append(0.0)
                    fr['occluded'].append(0)
                    alpha = cb[6] - np.arctan2(cb[0], cb[2])
                    fr['alpha'].append(alpha)
                    fr['bbox'].append(bb2d)
                    fr['dimensions'].append(cb[3:6])
                    fr['location'].append(cb[:3])
                    fr['rotation_y'].append(cb[6])
                    fr['score'].append(sc)

                write_label(fr, os.path.join(saved_submit, f'{idx:06d}.txt'))
                format_results[idx] = {k: np.array(v) for k, v in fr.items()}

        write_pickle(format_results, os.path.join(saved_path, 'results.pkl'))

    # fill missing indices
    all_ids = set(val_dataset.data_infos.keys())
    missing = all_ids - set(format_results.keys())
    for idx in missing:
        format_results[idx] = {
            'name':       np.array([]),
            'truncated':  np.array([]),
            'occluded':   np.array([]),
            'alpha':      np.array([]),
            'bbox':       np.empty((0, 4), dtype=np.float32),
            'dimensions': np.empty((0, 3), dtype=np.float32),
            'location':   np.empty((0, 3), dtype=np.float32),
            'rotation_y': np.empty((0, 1), dtype=np.float32),
            'score':      np.array([])
        }

    print('Evaluating.. Please wait several seconds.')
    do_eval(format_results, val_dataset.data_infos, CLASSES, saved_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root',  default='/mnt/ssd1/lifa_rdata/det/kitti',
                        help='your data root for kitti')
    parser.add_argument('--ckpt',       default='pretrained/epoch_160.pth',
                        help='your checkpoint for kitti')
    parser.add_argument('--saved_path', default='results',
                        help='your saved path for predicted results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument('--nclasses',   type=int, default=3)
    parser.add_argument('--no_cuda',    action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
