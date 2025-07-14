# loss/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb # Keep pdb if you use it for debugging

class Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=1/9, cls_w=1.0, reg_w=2.0, dir_w=0.2):
        super().__init__()
        self.alpha = alpha # Focal loss alpha parameter
        self.gamma = gamma # Focal loss gamma parameter
        self.cls_w = cls_w # Weight for classification loss
        self.reg_w = reg_w # Weight for regression loss
        self.dir_w = dir_w # Weight for direction loss

        # Regression Loss (Smooth L1) - reduction='none' to allow manual normalization
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none', beta=beta)

        # Direction Classification Loss (Cross Entropy) - reduction='none' for manual normalization
        # It expects raw logits as input.
        self.dir_cls_loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self,
                bbox_cls_pred,      # Raw logits from head (N, num_classes) e.g., (N, 3)
                bbox_pred,          # Regression predictions (N_pos, 7)
                bbox_dir_cls_pred,  # Direction logits (N_pos, 2)
                batched_labels,     # Class labels after masking by cls_mask (N,) -> contains values 0, 1, 2, ..., nclasses (background/remapped ignore)
                num_cls_pos,        # Scalar count of positive anchors (where label < nclasses) - should be python int from train_new.py
                batched_bbox_reg,   # Regression targets (N_pos, 7)
                batched_dir_labels):# Direction targets (N_pos,) -> contains values 0 or 1

        loss_dict = {}
        # Ensure num_cls_pos is a float tensor for division, clamp for stability
        # Convert the incoming Python int to a float tensor on the correct device
        num_pos_tensor = torch.tensor(num_cls_pos, device=bbox_cls_pred.device, dtype=torch.float32)
        num_pos_normalizer = num_pos_tensor.clamp(min=1.0) # Use this for normalization

        # --- 1. Classification Loss (Focal Loss with BCEWithLogitsLoss) ---
        cls_loss = torch.tensor(0.0, device=bbox_cls_pred.device, dtype=torch.float32) # Store final loss as float32

        if bbox_cls_pred.numel() > 0 and batched_labels.numel() > 0:
            num_classes = bbox_cls_pred.shape[-1]

            # Ensure labels are 1D long type
            if batched_labels.dim() != 1:
                 print(f"WARNING loss.py: Received non-1D batched_labels (shape: {batched_labels.shape}). Reshaping.")
                 batched_labels = batched_labels.reshape(-1)
            batched_labels_long = batched_labels.long()

            # Create one-hot targets [N, num_classes]
            target_probs = torch.zeros_like(bbox_cls_pred) # Inherits dtype (e.g., float16)

            pos_label_mask = (batched_labels_long >= 0) & (batched_labels_long < num_classes)
            valid_labels = batched_labels_long[pos_label_mask]
            scatter_indices = torch.where(pos_label_mask)[0]

            #if (batched_labels_long >= num_classes).any():
                 #print(f"WARNING loss.py: Label index >= num_classes detected ({batched_labels_long.max()}). These should be ignored by pos_label_mask.")

            if valid_labels.numel() > 0:
                 if len(scatter_indices) == len(valid_labels):
                      # Create one-hot and cast to target dtype
                      one_hot_targets = F.one_hot(valid_labels, num_classes=num_classes).to(target_probs.dtype)
                      target_probs.index_put_((scatter_indices,), one_hot_targets)
                 else:
                      print(f"ERROR loss.py: Mismatch scatter_indices/valid_labels in cls loss.")

            # Calculate BCE loss per element using logits (safe for autocast)
            if bbox_cls_pred.shape == target_probs.shape:
                 bce_loss_unreduced = F.binary_cross_entropy_with_logits(
                     bbox_cls_pred,
                     target_probs,
                     reduction='none'
                 )

                 # Calculate Focal Loss weights
                 with torch.no_grad():
                     p = torch.sigmoid(bbox_cls_pred)
                     p_t = torch.where(target_probs == 1.0, p, 1.0 - p)
                     alpha_t = torch.where(target_probs == 1.0, self.alpha, 1.0 - self.alpha)
                     focal_weights = alpha_t * torch.pow(1.0 - p_t, self.gamma)

                 # >>>>>>>>>> FIX: Apply weights <<<<<<<<<<<<<<
                 weighted_bce_loss = bce_loss_unreduced * focal_weights
                 # >>>>>>>>>>>>>> END FIX <<<<<<<<<<<<<<<<<<<<

                 # Sum the weighted loss and normalize by the number of positive anchors
                 cls_loss = weighted_bce_loss.sum() / num_pos_normalizer

            else:
                 print(f"ERROR loss.py: Shape mismatch just before BCE! Pred: {bbox_cls_pred.shape}, Target: {target_probs.shape}")
                 # cls_loss remains 0.0


        loss_dict['cls_loss'] = cls_loss.to(torch.float32) * self.cls_w


        # --- 2. Regression Loss (Smooth L1) ---
        reg_loss = torch.tensor(0.0, device=bbox_pred.device, dtype=torch.float32) # Store final loss as float32

        if bbox_pred.numel() > 0 and batched_bbox_reg.numel() > 0:
            if bbox_pred.shape == batched_bbox_reg.shape:
                 reg_loss_unreduced = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
                 reg_loss_sum_per_sample = reg_loss_unreduced.sum(dim=-1)
                 reg_loss = reg_loss_sum_per_sample.sum() / num_pos_normalizer
            else:
                 print(f"ERROR loss.py: Shape mismatch for regression! Pred: {bbox_pred.shape}, Target: {batched_bbox_reg.shape}")

        loss_dict['reg_loss'] = reg_loss.to(torch.float32) * self.reg_w


        # --- 3. Direction Classification Loss (Cross Entropy) ---
        dir_cls_loss = torch.tensor(0.0, device=bbox_dir_cls_pred.device, dtype=torch.float32) # Store final loss as float32

        if bbox_dir_cls_pred.numel() > 0 and batched_dir_labels.numel() > 0:
             dir_target_long = batched_dir_labels.long()
             if bbox_dir_cls_pred.shape[0] == dir_target_long.shape[0]:
                  dir_loss_unreduced = self.dir_cls_loss_func(bbox_dir_cls_pred, dir_target_long)
                  dir_cls_loss = dir_loss_unreduced.sum() / num_pos_normalizer
             else:
                  print(f"ERROR loss.py: Shape mismatch for direction! Pred: {bbox_dir_cls_pred.shape}, Target: {dir_target_long.shape}")

        loss_dict['dir_cls_loss'] = dir_cls_loss.to(torch.float32) * self.dir_w


        # --- 4. Total Loss ---
        total_loss = loss_dict['cls_loss'] + loss_dict['reg_loss'] + loss_dict['dir_cls_loss']
        loss_dict['total_loss'] = total_loss # Already float32

        return loss_dict