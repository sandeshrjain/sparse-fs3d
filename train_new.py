import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import logging
import csv
from utils import setup_seed
from dataset import Kitti, get_dataloader
from model import PointPillars  # Assuming this imports your softvpillars_v2.py PointPillars
from loss import Loss
from evaluate import main as eval_main
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

# Suppress the specific UserWarning about torch.meshgrid indexing argument if desired
# import warnings
# warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True) # Enable only if debugging gradient issues, slows down training

def log_metrics(loss_dict, global_step, phase):
    # Safely format loss values, handling potential non-numeric entries if any issue occurs
    loss_items = []
    for k, v in loss_dict.items():
        try:
            loss_items.append(f"{k}: {float(v):.6f}")
        except (ValueError, TypeError):
             loss_items.append(f"{k}: {v}") # Log as is if conversion fails
    msg = f"Step {global_step}, Phase: {phase}, " + ", ".join(loss_items)
    logging.info(msg)

def save_metrics_to_csv(loss_dict, global_step, phase, csv_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Check if file exists to write header
    file_exists = os.path.isfile(csv_path)
    
    # Define headers based on the first loss_dict encountered or predefined
    # Assuming standard keys: 'cls_loss', 'reg_loss', 'dir_cls_loss', 'total_loss'
    headers = ['global_step', 'phase', 'cls_loss', 'reg_loss', 'dir_cls_loss', 'total_loss']
    
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header if file is new
        if not file_exists or os.path.getsize(csv_path) == 0:
             writer.writerow(headers)
             
        # Prepare row data, ensuring order matches headers
        row = [global_step, phase]
        for key in headers[2:]: # Skip global_step and phase
             row.append(loss_dict.get(key, 'N/A')) # Use .get for safety
             
        writer.writerow(row)

def main(args):
    setup_seed()
    os.makedirs(args.saved_path, exist_ok=True)
    log_file_path = os.path.join(args.saved_path, "training_log.txt")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler() # Also log to console
        ]
    )
    logging.info("Starting training script...")
    logging.info(f"Arguments: {args}")

    device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Use torch.amp.autocast for modern PyTorch versions
    use_amp = not args.no_cuda and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    logging.info(f"Automatic Mixed Precision (AMP) enabled: {use_amp}")

    # Data
    logging.info("Loading datasets...")
    train_ds = Kitti(args.data_root, split='train')
    val_ds   = Kitti(args.data_root, split='val')
    train_loader = get_dataloader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader   = get_dataloader(val_ds,   args.batch_size, args.num_workers, shuffle=False)
    logging.info(f"Train dataset size: {len(train_ds)}, Val dataset size: {len(val_ds)}")
    logging.info(f"Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}")


    # Model
    logging.info(f"Initializing PointPillars model with {args.nclasses} classes...")
    model = PointPillars(nclasses=args.nclasses).to(device)
    
    # --- Determine num_anchors_per_loc based on model's head ---
    # This assumes the head structure is consistent
    # Example: head output channels = num_anchors * num_features (cls=nclasses, reg=7, dir=2)
    # Find one of the head conv layers to infer num_anchors
    try:
        head_cls_channels = model.head.conv_cls.out_channels
        num_anchors_per_loc = head_cls_channels // args.nclasses
        logging.info(f"Inferred num_anchors_per_loc: {num_anchors_per_loc}")
    except AttributeError:
        logging.error("Could not automatically infer num_anchors_per_loc from model head. Please check model structure.")
        # Provide a default or raise error if critical
        num_anchors_per_loc = 2 * args.nclasses # Fallback based on previous context
        logging.warning(f"Falling back to num_anchors_per_loc = {num_anchors_per_loc}")


    if torch.cuda.device_count() > 1 and not args.no_cuda:
        logging.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    # Resume
    start_epoch = 0
    ckpt_dir = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.resume_ckpt:
        # Try to construct full path if just filename is given
        if not os.path.isfile(args.resume_ckpt):
             ckpt_path = os.path.join(ckpt_dir, args.resume_ckpt)
        else:
             ckpt_path = args.resume_ckpt
             
        if os.path.isfile(ckpt_path):
            logging.info(f"Resuming training from checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            
            # Handle loading state dict for DataParallel or single model
            model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
            
            # Adjust state dict keys if necessary (e.g., remove 'module.' prefix)
            if isinstance(model, torch.nn.DataParallel) and not list(state.keys())[0].startswith('module.'):
                 state = {'module.' + k: v for k, v in state.items()}
            elif not isinstance(model, torch.nn.DataParallel) and list(state.keys())[0].startswith('module.'):
                 state = {k.replace('module.', '', 1): v for k, v in state.items()}
            
            try:
                 model_to_load.load_state_dict(state)
                 # Try to infer epoch from filename
                 try:
                      start_epoch = int(os.path.basename(args.resume_ckpt).split('_')[-1].split('.')[0])
                      logging.info(f"Resuming from epoch {start_epoch}")
                 except:
                      logging.warning("Could not infer epoch from checkpoint filename, starting epoch count from 0 for logging purposes.")
                      start_epoch = 0 # Reset epoch count if inference fails
            except RuntimeError as e:
                 logging.error(f"Error loading state dict: {e}. Model architecture might have changed.")
                 # Optionally exit or continue with fresh weights
                 # exit(1)
        else:
            logging.warning(f"Resume checkpoint not found at {ckpt_path}. Starting training from scratch.")

    # Loss / Optim / Scheduler
    loss_func = Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, betas=(0.95,0.99), weight_decay=args.weight_decay) # Added weight decay arg
    
    # Calculate total_iters, handle potential case of empty loader
    if len(train_loader) == 0:
         logging.warning("Training loader is empty. Setting total_iters to 1 to avoid scheduler errors.")
         total_iters = 1
    else:
         total_iters = len(train_loader) * args.max_epoch
         
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.init_lr * args.lr_scale_factor, # Use scale factor arg
        total_steps=total_iters,
        pct_start=0.4,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.95 * 0.895,
        max_momentum=0.95,
        div_factor=args.lr_div_factor # Use div factor arg
    )
    logging.info(f"Optimizer: AdamW, Initial LR: {args.init_lr}, Max LR: {args.init_lr * args.lr_scale_factor}")
    logging.info(f"Scheduler: OneCycleLR, Total Steps: {total_iters}")


    # CSV Logging Setup
    csv_log_path = os.path.join(args.saved_path, 'training_log.csv')
    logging.info(f"Logging metrics to CSV: {csv_log_path}")
    # Header is written inside save_metrics_to_csv if file is new

    # Training loop
    logging.info("Starting training loop...")
    global_step = start_epoch * len(train_loader) if len(train_loader) > 0 else 0

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        train_epoch_losses = {k: 0.0 for k in ['cls_loss', 'reg_loss', 'dir_cls_loss', 'total_loss']}
        
        # Use tqdm context manager for proper cleanup
        with tqdm(train_loader, desc=f'Train Epoch {epoch}/{args.max_epoch-1}') as pbar:
            for i, data in enumerate(pbar):
                # Move tensors to device
                try:
                    for k, v in data.items():
                         # Ensure v is a list/tuple before iterating
                         if isinstance(v, (list, tuple)):
                              data[k] = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in v]
                         elif torch.is_tensor(v): # Handle cases where data items might not be lists
                              data[k] = v.to(device, non_blocking=True)
                except Exception as e:
                     logging.error(f"Error moving data to device at step {i}: {e}")
                     continue # Skip batch if error occurs

                pts, gt_b, gt_l = data['batched_pts'], data['batched_gt_bboxes'], data['batched_labels']
                
                # Check if input lists are empty (can happen with small datasets/filtering)
                if not pts or not gt_b or not gt_l:
                    logging.warning(f"Skipping batch {i} due to empty input data.")
                    continue
                    
                optimizer.zero_grad(set_to_none=True) # More memory efficient

                try:
                    # AMP context manager
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        # Model forward pass
                        cls_p, reg_p, dir_p, at = model(
                            batched_pts=pts,
                            mode='train',
                            batched_gt_bboxes=gt_b,
                            batched_gt_labels=gt_l
                        )

                        # Get shapes
                        B, C_cls, H, W = cls_p.shape
                        C_reg = reg_p.shape[1]
                        C_dir = dir_p.shape[1]

                        # --- Correctly reshape predictions ---
                        cls_flat = cls_p.permute(0, 2, 3, 1).reshape(B, H, W, num_anchors_per_loc, args.nclasses).reshape(-1, args.nclasses)
                        reg_flat = reg_p.permute(0, 2, 3, 1).reshape(B, H, W, num_anchors_per_loc, 7).reshape(-1, 7)
                        dir_flat = dir_p.permute(0, 2, 3, 1).reshape(B, H, W, num_anchors_per_loc, 2).reshape(-1, 2)

                        # --- Targets ---
                        labels     = at['batched_labels'].reshape(-1)
                        lw         = at['batched_label_weights'].reshape(-1)
                        reg_target = at['batched_bbox_reg'].reshape(-1, 7)
                        dir_target = at['batched_dir_labels'].reshape(-1)

                        # Ensure consistency before masking
                        num_preds = cls_flat.shape[0]
                        num_targets = labels.shape[0]
                        if num_preds != num_targets:
                            logging.error(f"Shape mismatch! Flattened preds ({num_preds}) != Flattened labels ({num_targets})")
                            logging.error(f"cls_p shape: {cls_p.shape}, at['batched_labels'] shape: {at['batched_labels'].shape}")
                            # This often indicates an issue with anchor generation or target assignment
                            continue # Skip batch

                        # --- Classification Masking ---
                        # Mask includes positive (1) and negative (0) anchors, excludes ignore (-1)
                        cls_mask = lw >= 0
                        cls_input  = cls_flat[cls_mask]
                        cls_target = labels[cls_mask].clone() # Clone to avoid modifying original labels
                        # Remap ignore labels (which shouldn't be selected by cls_mask >= 0, but added safety)
                        # to the background class index ONLY AFTER masking
                        cls_target[cls_target < 0] = args.nclasses

                        # --- Regression/Direction Masking ---
                        # Mask includes only positive anchors (class 0 to nclasses-1)
                        pos_mask = (labels >= 0) & (labels < args.nclasses)
                        
                        # Apply positive mask to flattened predictions and targets
                        reg_input = reg_flat[pos_mask]
                        reg_target_masked = reg_target[pos_mask]
                        dir_input = dir_flat[pos_mask]
                        dir_target_masked = dir_target[pos_mask]

                        num_pos = pos_mask.sum().clamp(min=1).item() # Get scalar value

                        # --- Angle wrapping for regression loss ---
                        # Apply only if there are positive examples
                        if reg_input.shape[0] > 0:
                            yaw_p = reg_input[:, -1] # No need to clone if only used for calculation
                            yaw_t = reg_target_masked[:, -1]
                            # Modify reg_input in-place for loss calculation
                            reg_input[:, -1] = torch.sin(yaw_p - yaw_t)
                            # Note: Some loss functions might expect the raw diff (yaw_p - yaw_t)
                            # or handle sin/cos internally. Adjust if needed based on your Loss class.
                            # Example using sin/cos difference directly:
                            # reg_input[:, -1] = torch.sin(yaw_p)*torch.cos(yaw_t) - torch.cos(yaw_p)*torch.sin(yaw_t)


                        # --- Calculate Loss ---
                        loss_d = loss_func(
                            bbox_cls_pred=cls_input,
                            bbox_pred=reg_input,            # Use modified reg_input
                            bbox_dir_cls_pred=dir_input,
                            batched_labels=cls_target,      # Masked & remapped labels
                            num_cls_pos=num_pos,
                            batched_bbox_reg=reg_target_masked, # Masked reg targets
                            batched_dir_labels=dir_target_masked # Masked dir targets
                        )
                        
                        # Check for NaN/Inf loss
                        total_loss = loss_d['total_loss']
                        if torch.isnan(total_loss) or torch.isinf(total_loss):
                             logging.warning(f"NaN or Inf loss detected at step {global_step}. Skipping backward pass. Losses: {loss_d}")
                             # Optionally save problematic batch data for debugging
                             # torch.save(data, f"problem_batch_step_{global_step}.pt")
                             continue # Skip backward/step


                    # --- Backward pass and optimizer step ---
                    scaler.scale(total_loss).backward()
                    # Optional: Gradient clipping
                    # scaler.unscale_(optimizer) # Unscale first if needed by clipping
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=...)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # --- Logging and Accumulation ---
                    current_losses = {k: v.item() for k, v in loss_d.items()}
                    for k in train_epoch_losses:
                        train_epoch_losses[k] += current_losses.get(k, 0.0)

                    # Update progress bar description
                    pbar.set_postfix({
                        'loss': f"{current_losses['total_loss']:.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })

                    if global_step % args.log_freq == 0:
                        log_metrics(current_losses, global_step, 'train')
                        save_metrics_to_csv(current_losses, global_step, 'train', csv_log_path)

                    global_step += 1

                except Exception as e:
                    logging.exception(f"Error during training step {i} in epoch {epoch}: {e}")
                    # Optionally: try to continue to next batch or re-raise
                    raise e # Uncomment to stop training on error


        # --- End of Epoch ---
        avg_train_losses = {k: train_epoch_losses[k] / len(train_loader) if len(train_loader) > 0 else 0.0 for k in train_epoch_losses}
        logging.info(f"Epoch {epoch} Average Train Losses: {avg_train_losses}")

        # --- Save Checkpoint ---
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            ckpt_save_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth')
            logging.info(f"Saving checkpoint to {ckpt_save_path}")
            # Save model state dict correctly for DP or single model
            model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(model_state, ckpt_save_path)

        # --- Validation ---
        # Skip validation every N epochs if desired (e.g., validation is slow)
        if epoch % args.val_freq_epoch != 0 and epoch != args.max_epoch - 1: # Validate less frequently, but always on last epoch
             logging.info(f"Skipping validation for epoch {epoch}.")
             continue

        model.eval()
        val_epoch_losses = {k: 0.0 for k in train_epoch_losses}
        logging.info(f"Starting validation for epoch {epoch}...")

        with torch.no_grad():
            with tqdm(val_loader, desc=f'Val Epoch {epoch}/{args.max_epoch-1}') as pbar_val:
                 for i_val, data_val in enumerate(pbar_val):
                    # Move tensors to device
                    try:
                         for k, v in data_val.items():
                              if isinstance(v, (list, tuple)):
                                   data_val[k] = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in v]
                              elif torch.is_tensor(v):
                                   data_val[k] = v.to(device, non_blocking=True)
                    except Exception as e:
                         logging.error(f"Error moving validation data to device at step {i_val}: {e}")
                         continue # Skip batch


                    pts_val, gt_b_val, gt_l_val = data_val['batched_pts'], data_val['batched_gt_bboxes'], data_val['batched_labels']
                    
                    if not pts_val or not gt_b_val or not gt_l_val:
                         logging.warning(f"Skipping validation batch {i_val} due to empty input data.")
                         continue
                         
                    try:
                        # Run model in 'train' mode to get anchor targets for loss calculation
                        # Ensure AMP is used consistently if enabled
                        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                            cls_p_val, reg_p_val, dir_p_val, at_val = model(
                                batched_pts=pts_val,
                                mode='train', # Get targets for loss calc
                                batched_gt_bboxes=gt_b_val,
                                batched_gt_labels=gt_l_val
                            )

                            # Get shapes
                            B_val, C_cls_val, H_val, W_val = cls_p_val.shape

                            # --- Correctly reshape predictions ---
                            cls_flat_val = cls_p_val.permute(0, 2, 3, 1).reshape(B_val, H_val, W_val, num_anchors_per_loc, args.nclasses).reshape(-1, args.nclasses)
                            reg_flat_val = reg_p_val.permute(0, 2, 3, 1).reshape(B_val, H_val, W_val, num_anchors_per_loc, 7).reshape(-1, 7)
                            dir_flat_val = dir_p_val.permute(0, 2, 3, 1).reshape(B_val, H_val, W_val, num_anchors_per_loc, 2).reshape(-1, 2)

                            # --- Targets ---
                            labels_val     = at_val['batched_labels'].reshape(-1)
                            lw_val         = at_val['batched_label_weights'].reshape(-1)
                            reg_target_val = at_val['batched_bbox_reg'].reshape(-1, 7)
                            dir_target_val = at_val['batched_dir_labels'].reshape(-1)
                            
                            # --- Shape Consistency Check ---
                            num_preds_val = cls_flat_val.shape[0]
                            num_targets_val = labels_val.shape[0]
                            if num_preds_val != num_targets_val:
                                logging.error(f"[Validation] Shape mismatch! Flattened preds ({num_preds_val}) != Flattened labels ({num_targets_val})")
                                continue # Skip batch

                            # --- Classification Masking ---
                            cls_mask_val = lw_val >= 0
                            cls_input_val  = cls_flat_val[cls_mask_val]
                            cls_target_val = labels_val[cls_mask_val].clone()
                            cls_target_val[cls_target_val < 0] = args.nclasses

                            # --- Regression/Direction Masking ---
                            pos_mask_val = (labels_val >= 0) & (labels_val < args.nclasses)
                            reg_input_val = reg_flat_val[pos_mask_val]
                            reg_target_masked_val = reg_target_val[pos_mask_val]
                            dir_input_val = dir_flat_val[pos_mask_val]
                            dir_target_masked_val = dir_target_val[pos_mask_val]
                            num_pos_val = pos_mask_val.sum().clamp(min=1).item()

                            # --- Angle wrapping ---
                            if reg_input_val.shape[0] > 0:
                                yaw_p_val = reg_input_val[:, -1]
                                yaw_t_val = reg_target_masked_val[:, -1]
                                reg_input_val[:, -1] = torch.sin(yaw_p_val - yaw_t_val) # Or sin/cos diff

                            # --- Calculate Loss ---
                            loss_v = loss_func(
                                bbox_cls_pred=cls_input_val,
                                bbox_pred=reg_input_val,
                                bbox_dir_cls_pred=dir_input_val,
                                batched_labels=cls_target_val,
                                num_cls_pos=num_pos_val,
                                batched_bbox_reg=reg_target_masked_val,
                                batched_dir_labels=dir_target_masked_val
                            )
                            
                        # Accumulate validation losses
                        current_val_losses = {k: v.item() for k, v in loss_v.items()}
                        for k in val_epoch_losses:
                            val_epoch_losses[k] += current_val_losses.get(k, 0.0)
                            
                        pbar_val.set_postfix({'loss': f"{current_val_losses['total_loss']:.4f}"})

                    except Exception as e:
                         logging.exception(f"Error during validation step {i_val} in epoch {epoch}: {e}")
                         # Optionally continue


        avg_val_losses = {k: val_epoch_losses[k] / len(val_loader) if len(val_loader) > 0 else 0.0 for k in val_epoch_losses}
        logging.info(f"Epoch {epoch} Average Validation Losses: {avg_val_losses}")
        # Log validation losses to CSV with a representative step number for the epoch end
        save_metrics_to_csv(avg_val_losses, global_step, 'val_epoch_avg', csv_log_path)


        # --- Optional: Run full evaluation script ---
        if args.run_eval and (epoch + 1) % args.eval_freq_epoch == 0: # Added eval frequency arg
            ckpt_to_eval = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth')
            if os.path.exists(ckpt_to_eval):
                logging.info(f"Running full evaluation for epoch {epoch+1} checkpoint...")
                eval_output_path = os.path.join(args.saved_path, f'eval_results_epoch_{epoch+1}')
                eval_args = argparse.Namespace(
                    data_root=args.data_root,
                    ckpt=ckpt_to_eval,
                    saved_path=eval_output_path, # Use specific output path
                    batch_size=args.batch_size, # Use training batch size or specify different eval batch size
                    num_workers=args.num_workers,
                    nclasses=args.nclasses,
                    no_cuda=args.no_cuda
                    # Add any other args needed by evaluate.py
                )
                try:
                    eval_main(eval_args) # Call the main function from evaluate.py
                except Exception as e:
                    logging.exception(f"Error during evaluation script run for epoch {epoch+1}: {e}")
            else:
                logging.warning(f"Checkpoint {ckpt_to_eval} not found, skipping evaluation.")
                
    logging.info("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PointPillars Training Script")
    
    # Data args
    parser.add_argument('--data_root', type=str, default='/vtti/scratch/sjain/PointPillars/dataset/kitti', help='Path to KITTI dataset root')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU') # Increased default slightly
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers')

    # Model args
    parser.add_argument('--nclasses', type=int, default=3, help='Number of classes (e.g., Car, Pedestrian, Cyclist)')

    # Training args
    parser.add_argument('--max_epoch', type=int, default=160, help='Total number of epochs to train') # Adjusted default common for Kitti
    parser.add_argument('--init_lr', type=float, default=0.00025, help='Initial learning rate for AdamW')
    parser.add_argument('--lr_scale_factor', type=float, default=10.0, help='Max LR factor for OneCycleLR (max_lr = init_lr * factor)')
    parser.add_argument('--lr_div_factor', type=float, default=10.0, help='Initial LR divisor for OneCycleLR (start_lr = max_lr / div_factor)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    
    # Logging/Checkpointing args
    parser.add_argument('--saved_path', type=str, default='pillar_logs_soft_v2', help='Directory to save logs and checkpoints')
    parser.add_argument('--log_freq', type=int, default=50, help='Frequency (in steps) to log training metrics')
    parser.add_argument('--ckpt_freq_epoch', type=int, default=5, help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--val_freq_epoch', type=int, default=5, help='Frequency (in epochs) to run validation loss calculation')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Path or filename of checkpoint to resume from (in saved_path/checkpoints)')
    
    # Evaluation args
    parser.add_argument('--run_eval', action='store_true', help='Run full evaluation script periodically')
    parser.add_argument('--eval_freq_epoch', type=int, default=1, help='Frequency (in epochs) to run full evaluation')

    # System args
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA training')

    args = parser.parse_args()
    
    # Simple validation for paths
    if not os.path.isdir(args.data_root):
        print(f"Error: Data root directory not found at {args.data_root}")
        exit(1)

    main(args)