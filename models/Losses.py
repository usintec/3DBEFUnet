import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, class_weights=None, num_classes=4, gamma=2.0, alpha=0.25):
        """
        Dice + Focal Loss for multi-class segmentation

        Args:
            dice_weight (float): weight for dice part
            focal_weight (float): weight for focal part
            class_weights (list or None): per-class weights [WT, TC, ET, BG] or None for uniform
            num_classes (int): number of segmentation classes (BraTS = 4 incl. background)
            gamma (float): focusing parameter for focal loss
            alpha (float or list): balance factor for focal loss
        """
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.num_classes = num_classes
        self.gamma = gamma

        if class_weights is None:
            class_weights = torch.ones(num_classes)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

        if isinstance(alpha, (float, int)):
            alpha = torch.ones(num_classes) * alpha
        else:
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): model outputs [B, C, H, W, D]
            targets (torch.Tensor): ground truth [B, H, W, D] with class indices
        """
        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)  # [B,H,W,D,C]
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()     # [B,C,H,W,D]

        probs = torch.softmax(logits, dim=1)  # [B,C,H,W,D]

        # -----------------------
        # Dice Loss
        # -----------------------
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        denominator = torch.sum(probs + targets_onehot, dims)
        dice_loss = 1.0 - (2. * intersection + 1e-5) / (denominator + 1e-5)
        dice_loss = torch.sum(self.class_weights * dice_loss) / torch.sum(self.class_weights)

        # -----------------------
        # Focal Loss
        # -----------------------
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')  # [B,H,W,D]
        pt = torch.exp(-ce_loss)  # pt = prob of true class
        focal_loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()

        # -----------------------
        # Combine
        # -----------------------
        loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        return loss

class BalancedLoss(nn.Module):
    def __init__(self, ce_weight=0.8, dice_weight=0.2, class_weights=None, num_classes=4):
        """
        Balanced loss = weighted CE + weighted Generalized Dice Loss

        Args:
            ce_weight (float): weight for cross entropy part
            dice_weight (float): weight for dice part
            class_weights (list or None): per-class weights [WT, TC, ET, Background?] 
                                          If None, all classes get equal weight.
            num_classes (int): number of segmentation classes (BraTS = 4 incl. background)
        """
        super(BalancedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if class_weights is None:
            self.class_weights = torch.ones(num_classes)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.num_classes = num_classes

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): model outputs [B, C, H, W, D]
            targets (torch.Tensor): ground truth [B, H, W, D] with class indices
        """
        device = logits.device
        self.class_weights = self.class_weights.to(device)

        # CrossEntropyLoss with class weights
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)

        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)  # [B,H,W,D,C]
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()     # [B,C,H,W,D]

        probs = torch.softmax(logits, dim=1)  # [B,C,H,W,D]

        # Generalized Dice Loss
        dims = (0, 2, 3, 4)  # reduce over batch + spatial dims
        intersection = torch.sum(probs * targets_onehot, dims)
        denominator = torch.sum(probs + targets_onehot, dims)

        gdl_loss = 1.0 - (2. * intersection + 1e-5) / (denominator + 1e-5)

        # Apply class weights to dice
        gdl_loss = torch.sum(self.class_weights * gdl_loss) / torch.sum(self.class_weights)

        # Combine
        loss = self.ce_weight * ce_loss + self.dice_weight * gdl_loss
        return loss

class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss (GDL) for multi-class segmentation.
    Handles class imbalance by weighting inversely proportional
    to squared class volume (Sudre et al., 2017).
    """

    def __init__(self, softmax: bool = True, ignore_index: int = None, eps: float = 1e-6):
        """
        Args:
            softmax (bool): Whether to apply softmax to logits.
            ignore_index (int): Class index to ignore in loss calculation.
            eps (float): Smoothing term for numerical stability.
        """
        super(GeneralizedDiceLoss, self).__init__()
        self.softmax = softmax
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W, D] raw logits or probabilities
            targets: [B, H, W, D] ground-truth segmentation
        Returns:
            loss (scalar tensor)
        """
        # Apply softmax to get class probabilities
        if self.softmax:
            inputs = F.softmax(inputs, dim=1)

        num_classes = inputs.shape[1]

        # One-hot encode targets
        targets_onehot = F.one_hot(targets.long(), num_classes=num_classes)  # [B, H, W, D, C]
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()       # [B, C, H, W, D]

        # Mask ignore_index voxels
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            mask = mask.unsqueeze(1)  # [B, 1, H, W, D]
            inputs = inputs * mask
            targets_onehot = targets_onehot * mask

        # Flatten spatial dims: [B, C, -1]
        inputs = inputs.contiguous().view(inputs.shape[0], num_classes, -1)
        targets_onehot = targets_onehot.contiguous().view(targets.shape[0], num_classes, -1)

        # Compute per-class volume
        w = 1.0 / (torch.pow(torch.sum(targets_onehot, dim=2), 2) + self.eps)  # [B, C]

        # Intersection and union
        intersect = torch.sum(inputs * targets_onehot, dim=2)  # [B, C]
        denominator = torch.sum(inputs + targets_onehot, dim=2)  # [B, C]

        # Generalized Dice Score
        numerator = 2 * torch.sum(w * intersect, dim=1)
        denom = torch.sum(w * denominator, dim=1) + self.eps
        dice_score = numerator / denom

        # Loss = 1 - mean Dice
        return 1.0 - dice_score.mean()

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
