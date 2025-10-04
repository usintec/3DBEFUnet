import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

# class BoundaryLoss(nn.Module):
#     """
#     Boundary Loss for segmentation (Kervadec et al., 2019)
#     Encourages better boundary alignment using signed distance maps (SDMs).

#     Args:
#         num_classes (int): number of classes (including background)
#         ignore_background (bool): whether to skip background during loss computation
#     """
#     def __init__(self, num_classes, ignore_background=True):
#         super(BoundaryLoss, self).__init__()
#         self.num_classes = num_classes
#         self.ignore_background = ignore_background

#     def forward(self, inputs, target):
#         """
#         Args:
#             inputs: (B, C, D, H, W) predicted logits or probabilities
#             target: (B, D, H, W) ground-truth integer labels
#         """
#         if inputs.shape[2:] != target.shape[1:]:
#             target = F.interpolate(
#                 target.unsqueeze(1).float(),
#                 size=inputs.shape[2:], 
#                 mode='nearest'
#             ).squeeze(1).long()

#         # Convert logits to probabilities
#         probs = torch.softmax(inputs, dim=1)

#         total_loss = 0.0
#         num_classes_used = 0

#         for c in range(self.num_classes):
#             if self.ignore_background and c == 0:
#                 continue

#             # Extract class mask
#             gt_c = (target == c).cpu().numpy().astype(np.uint8)

#             # Compute Signed Distance Map (foreground positive, background negative)
#             sdm = np.zeros_like(gt_c, dtype=np.float32)
#             for b in range(gt_c.shape[0]):
#                 posmask = gt_c[b].astype(bool)
#                 if posmask.any():
#                     negmask = ~posmask
#                     sdm_pos = distance(posmask)
#                     sdm_neg = distance(negmask)
#                     sdm[b] = sdm_neg - sdm_pos  # signed distance map
#             sdm = torch.from_numpy(sdm).to(inputs.device)

#             # Get softmax probability for this class
#             pc = probs[:, c, ...]

#             # Boundary loss = mean of |prob - boundary_distance|
#             boundary_loss = torch.mean(pc * sdm)
#             total_loss += boundary_loss
#             num_classes_used += 1

#         return total_loss / num_classes_used
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

def one_hot2dist(seg):
    """
    Convert binary segmentation to distance map.
    """
    seg = seg.astype(np.bool_)
    res = np.zeros_like(seg, dtype=np.float32)
    for c in range(seg.shape[0]):
        posmask = seg[c].astype(np.bool_)
        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

# class BoundaryLoss(nn.Module):
#     def __init__(self, num_classes=4):
#         super(BoundaryLoss, self).__init__()
#         self.num_classes = num_classes

#     def forward(self, logits, target):
#         """
#         logits: (B, C, D, H, W)
#         target: (B, D, H, W)
#         """
#         n, c, d, h, w = logits.shape
#         target_onehot = F.one_hot(target, num_classes=c).permute(0, 4, 1, 2, 3).float()
#         probs = F.softmax(logits, dim=1)

#         # Compute distance maps for each target
#         target_np = target_onehot.cpu().numpy()
#         dist_maps = np.zeros_like(target_np)
#         for i in range(n):
#             dist_maps[i] = one_hot2dist(target_np[i])
#         dist_maps = torch.from_numpy(dist_maps).to(logits.device)

#         # Multiply probabilities with distance maps (class-wise)
#         boundary_loss = torch.sum(probs * dist_maps, dim=1)
#         return boundary_loss.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

class BoundaryLoss(nn.Module):
    def __init__(self, num_classes):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes

    def one_hot_encode(self, target, num_classes):
        """Convert target tensor to one-hot format"""
        shape = target.shape
        one_hot = torch.zeros((shape[0], num_classes, *shape[1:]), device=target.device)
        return one_hot.scatter_(1, target.unsqueeze(1).long(), 1)

    def compute_sdf(self, mask):
        """Compute signed distance map for each binary mask"""
        mask = mask.cpu().numpy().astype(np.uint8)
        sdf = np.zeros_like(mask, dtype=np.float32)

        for b in range(mask.shape[0]):
            posmask = mask[b].astype(np.bool_)
            if posmask.any():
                negmask = ~posmask
                posdis = distance_transform_edt(negmask)
                negdis = distance_transform_edt(posmask)
                sdf[b] = posdis - negdis
        return torch.from_numpy(sdf).to(mask.device)

    def forward(self, pred, target):
        """
        pred: (B, C, H, W, D) — logits
        target: (B, H, W, D)
        """
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[2:], mode='nearest').squeeze(1)

        # One-hot encoding
        target_onehot = self.one_hot_encode(target, self.num_classes)

        # Softmax over classes
        pred_softmax = F.softmax(pred, dim=1)

        # Compute signed distance function for GT
        with torch.no_grad():
            sdf = torch.stack([self.compute_sdf(target_onehot[:, c]) for c in range(self.num_classes)], dim=1)

        # Boundary loss
        multipled = pred_softmax * sdf
        boundary_loss = multipled.abs().mean()

        return boundary_loss

class ClassWiseDiscriminativeLoss(nn.Module):
    def __init__(self, delta_var=0.5, delta_dist=1.5,
                 param_var=1.0, param_dist=1.0, param_reg=0.001,
                 ignore_index=0):
        """
        Class-wise Discriminative Loss for semantic segmentation.
        Adapted from instance-level DLF (De Brabandere et al., 2017).
        
        Args:
            delta_var: margin for intra-class variance
            delta_dist: margin for inter-class distance
            param_var: weight for variance term
            param_dist: weight for distance term
            param_reg: weight for regularization term
            ignore_index: label ID to ignore (e.g., background=0 in BraTS)
        """
        super(ClassWiseDiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.param_var = param_var
        self.param_dist = param_dist
        self.param_reg = param_reg
        self.ignore_index = ignore_index

    def forward(self, embeddings, class_labels):
        """
        Args:
            embeddings: [B, C, D, H, W] → embedding space
            class_labels: [B, D, H, W] → class IDs (e.g., 0=background, 1=ET, 2=TC, 3=WT)
        Returns:
            loss: scalar tensor
        """
        B, C, D, H, W = embeddings.size()
        embeddings = embeddings.permute(0, 2, 3, 4, 1).contiguous()  # [B, D, H, W, C]

        total_loss = 0.0
        for b in range(B):
            embed = embeddings[b]      # [D, H, W, C]
            labels = class_labels[b]   # [D, H, W]

            # 🔥 Downsample labels if they don't match embedding resolution
            if labels.shape != embed.shape[:3]:
                labels = torch.nn.functional.interpolate(
                    labels.unsqueeze(0).unsqueeze(0).float(),
                    size=embed.shape[:3],
                    mode='nearest'
                ).squeeze().long()

            unique_classes = labels.unique()
            unique_classes = unique_classes[unique_classes != self.ignore_index]
            if len(unique_classes) == 0:
                continue

            class_means = []
            var_loss = 0.0

            for cls in unique_classes:
                mask = (labels == cls)
                if mask.sum() == 0:
                    continue

                # embeddings of this class
                embed_masked = embed[mask]  # [N, C]
                mean = embed_masked.mean(dim=0)
                class_means.append(mean)

                # variance term
                dist = torch.norm(embed_masked - mean, dim=1)
                dist = torch.clamp(dist - self.delta_var, min=0.0) ** 2
                var_loss += dist.mean()

            var_loss /= len(unique_classes)

            # distance term (between class centers)
            dist_loss = 0.0
            if len(class_means) > 1:
                class_means = torch.stack(class_means)
                for i in range(len(class_means)):
                    for j in range(i + 1, len(class_means)):
                        dist = torch.norm(class_means[i] - class_means[j])
                        dist = torch.clamp(self.delta_dist - dist, min=0.0) ** 2
                        dist_loss += dist
                dist_loss /= (len(class_means) * (len(class_means) - 1))

            # regularization (keep embeddings close to origin)
            if len(class_means) > 1:
                reg_loss = torch.mean(torch.norm(class_means, dim=1))
            else:  # single tensor case
                reg_loss = torch.norm(class_means[0])

            total_loss += (self.param_var * var_loss +
                        self.param_dist * dist_loss +
                        self.param_reg * reg_loss)

        return total_loss / B
