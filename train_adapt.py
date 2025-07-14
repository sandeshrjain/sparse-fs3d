import argparse
import os
import torch
from tqdm import tqdm
import pdb
import numpy as np
from utils import setup_seed
from dataset import Kitti, get_dataloader
from model import PointPillars
from loss import Loss


import logging
import csv
import os
import json

from evaluate import main as eval_main
torch.backends.cudnn.benchmark = True

torch.autograd.set_detect_anomaly(True)

def log_metrics(loss_dict, global_step, phase):
    """Logs loss metrics to a text file."""
    log_msg = f"Step {global_step}, Phase: {phase}, " + ", ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])
    logging.info(log_msg)



def save_metrics_to_csv(loss_dict, global_step, phase, csv_path):
    """Saves metrics to a CSV file."""
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        row = [global_step, phase] + [loss_dict[k] for k in loss_dict.keys()]
        writer.writerow(row)

#from torch.utils.tensorboard import SummaryWriter


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)

def save_eval_metrics_to_csv(metrics_dict, epoch, csv_path):
    """Save evaluation metrics like AP3D, APBEV to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch'] + list(metrics_dict.keys()))
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch] + [metrics_dict[k] for k in metrics_dict.keys()])

def main(args):
    setup_seed()
    # Set up logging
    log_path = os.path.join(args.saved_path, "training_log.txt")
    os.makedirs(args.saved_path, exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")

    device = torch.device("cuda:0" if not args.no_cuda else "cpu")
    train_dataset = Kitti(data_root=args.data_root,
                          split='train')
    val_dataset = Kitti(data_root=args.data_root,
                        split='val')
    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    if not args.no_cuda:
        device = torch.device("cuda:0")
        pointpillars = PointPillars(nclasses=args.nclasses).to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            pointpillars = torch.nn.DataParallel(pointpillars)
    else:
        device = torch.device("cpu")
        pointpillars = PointPillars(nclasses=args.nclasses).to(device)


    loss_func = Loss()

    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=init_lr*10, 
                                                    total_steps=max_iters, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    #writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)
    eval_csv_path = os.path.join(args.saved_path, 'eval_metrics.csv')


    for epoch in range(args.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        
        train_step, val_step = 0, 0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = item.to(device)
                            #data_dict[key][j] = data_dict[key][j].cuda()
            
            optimizer.zero_grad()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                pointpillars(batched_pts=batched_pts, 
                             mode='train',
                             batched_gt_bboxes=batched_gt_bboxes, 
                             batched_gt_labels=batched_labels)
            
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
            # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
            
            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
            bbox_pred = bbox_pred[pos_idx]
            batched_bbox_reg = batched_bbox_reg[pos_idx]
            # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
            # Safely compute the rotation without modifying in-place
            sin_diff = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
            cos_diff = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])

            bbox_pred_rot = bbox_pred.clone()
            bbox_pred_rot[:, -1] = sin_diff

            batched_bbox_reg_rot = batched_bbox_reg.clone()
            batched_bbox_reg_rot[:, -1] = cos_diff



            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
            if num_cls_pos == 0:
                num_cls_pos = 1  # prevent div-by-zero

            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            # Safely compute the rotation without modifying in-place
            sin_diff = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
            cos_diff = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])

            bbox_pred_rot = bbox_pred.clone()
            bbox_pred_rot[:, -1] = sin_diff

            batched_bbox_reg_rot = batched_bbox_reg.clone()
            batched_bbox_reg_rot[:, -1] = cos_diff



            loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                  bbox_pred=bbox_pred,
                                  bbox_dir_cls_pred=bbox_dir_cls_pred,
                                  batched_labels=batched_bbox_labels, 
                                  num_cls_pos=num_cls_pos, 
                                  batched_bbox_reg=batched_bbox_reg, 
                                  batched_dir_labels=batched_dir_labels)
            
            loss = loss_dict['total_loss']
            loss.backward()
            # for name, param in pointpillars.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.device)

            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            optimizer.step()
            scheduler.step()
            train_loss_accumulator = {key: 0.0 for key in loss_dict.keys()}

            # Create CSV for logging
            csv_path = os.path.join(args.saved_path, "training_log.csv")
            csv_fields = ["global_step", "phase"] + list(loss_dict.keys())

            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_fields)  # Write header

            global_step = epoch * len(train_dataloader) + train_step + 1


            for key in loss_dict:
                train_loss_accumulator[key] += loss_dict[key].item()

            if global_step % args.log_freq == 0:
                log_metrics(loss_dict, global_step, "train")
                save_metrics_to_csv(loss_dict, global_step, "train", csv_path)

            # if global_step % args.log_freq == 0:
            #     save_summary(loss_dict, global_step, 'train',
            #                  lr=optimizer.param_groups[0]['lr'], 
            #                  momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1

        # Compute average loss over the epoch
        avg_train_loss = {k: v / len(train_dataloader) for k, v in train_loss_accumulator.items()}

        # Print Training Stats for this epoch
        print(f"\n[TRAIN] Epoch {epoch+1} Summary:")
        print(" | ".join([f"{k}: {v:.6f}" for k, v in avg_train_loss.items()]))

        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            torch.save(pointpillars.state_dict(), os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))

        if epoch % 1 == 0:
            continue
        pointpillars.eval()
        val_loss_accumulator = {key: 0.0 for key in loss_dict.keys()}

        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                if not args.no_cuda:
                    # move the tensors to the cuda
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = item.to(device)
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                batched_difficulty = data_dict['batched_difficulty']
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                    pointpillars(batched_pts=batched_pts, 
                                mode='train',
                                batched_gt_bboxes=batched_gt_bboxes, 
                                batched_gt_labels=batched_labels)
                
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                if num_cls_pos == 0:
                    num_cls_pos = 1  # prevent div-by-zero

                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]
                bbox_pred[:, -1] = torch.clamp(bbox_pred[:, -1], min=-np.pi, max=np.pi)
                batched_bbox_reg[:, -1] = torch.clamp(batched_bbox_reg[:, -1], min=-np.pi, max=np.pi)

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                    bbox_pred=bbox_pred,
                                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                                    batched_labels=batched_bbox_labels, 
                                    num_cls_pos=num_cls_pos, 
                                    batched_bbox_reg=batched_bbox_reg, 
                                    batched_dir_labels=batched_dir_labels)
                
                global_step = epoch * len(val_dataloader) + val_step + 1

                for key in loss_dict:
                    val_loss_accumulator[key] += loss_dict[key].item()

                # if global_step % args.log_freq == 0:
                #     save_summary(writer, loss_dict, global_step, 'val')
                if global_step % args.log_freq == 0:
                    log_metrics(loss_dict, global_step, "val")
                    save_metrics_to_csv(loss_dict, global_step, "val", csv_path)
                val_step += 1

        # Compute average loss over the validation epoch
        avg_val_loss = {k: v for k, v in val_loss_accumulator.items()}

        # Print Validation Stats for this epoch
        print(f"\n[VALIDATION] Epoch {epoch+1} Summary:")
        print(" | ".join([f"{k}: {v:.6f}" for k, v in avg_val_loss.items()]))
        #s sff aaw aaa a 
        pointpillars.train()

        eval_args = argparse.Namespace(
            data_root=args.data_root,
            ckpt=os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'),
            saved_path=os.path.join(args.saved_path, f'results_epoch_{epoch+1}'),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            nclasses=args.nclasses,
            no_cuda=args.no_cuda
        )
        eval_main(eval_args)
        eval_file = os.path.join(args.saved_path, f'results_epoch_{epoch+1}', 'eval_results.txt')
        eval_metrics = {}
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, val = line.strip().split(':')[-2:]
                        try:
                            eval_metrics[key.strip()] = float(val.strip())
                        except:
                            continue
                    if 'Overall' in line or 'AP' in line:
                        logging.info(f'[EVAL Epoch {epoch+1}] {line.strip()}')
            save_eval_metrics_to_csv(eval_metrics, epoch+1, eval_csv_path)
            print(f"[EVAL CSV] Epoch {epoch+1} metrics saved to {eval_csv_path}")


    # After final epoch:
    best_ap3d = 0
    best_epoch = 0
    for e in range(args.max_epoch):
        eval_file = os.path.join(args.saved_path, f'results_epoch_{e+1}', 'eval_results.txt')
        if not os.path.exists(eval_file):
            continue
        with open(eval_file, 'r') as f:
            for line in f:
                if 'AP3D@0.7' in line and 'Car' in line:
                    ap = float(line.strip().split()[-1])
                    if ap > best_ap3d:
                        best_ap3d = ap
                        best_epoch = e+1

    print(f"\n Best Car AP3D@0.7: {best_ap3d:.2f} @ Epoch {best_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00015)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
