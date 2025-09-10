import torch
import torch.nn as nn


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

            # 🔥 Fix: downsample labels if resolution mismatches
            if labels.shape != embed.shape[:3]:
                labels = torch.nn.functional.interpolate(
                    labels.unsqueeze(0).unsqueeze(0).float(),
                    size=embed.shape[:3],  # match embedding [D,H,W]
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
            reg_loss = torch.mean(torch.norm(torch.stack(class_means), dim=1))

            total_loss += (self.param_var * var_loss +
                        self.param_dist * dist_loss +
                        self.param_reg * reg_loss)

        return total_loss / B
