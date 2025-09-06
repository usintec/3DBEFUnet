import torch
import torch.nn as nn
from sklearn.cluster import MeanShift as SkMeanShift


class MeanShiftClustering(nn.Module):
    def __init__(self, bandwidth=0.5, max_iter=300):
        """
        Mean Shift Clustering for embeddings.
        Args:
            bandwidth: kernel bandwidth
            max_iter: maximum iterations
        """
        super(MeanShiftClustering, self).__init__()
        self.bandwidth = bandwidth
        self.max_iter = max_iter

    def forward(self, embeddings, seg_logits):
        """
        Apply mean shift clustering to embeddings to refine segmentation.
        Args:
            embeddings: [B, C, D, H, W]
            seg_logits: [B, n_classes, D, H, W]
        Returns:
            refined_seg: [B, n_classes, D, H, W]
        """
        B, C, D, H, W = embeddings.size()
        refined_segs = []

        for b in range(B):
            emb = embeddings[b].view(C, -1).transpose(0, 1).detach().cpu().numpy()  # [N, C]
            seg = seg_logits[b].view(seg_logits.size(1), -1).detach().cpu().numpy()  # [n_classes, N]

            try:
                ms = SkMeanShift(bandwidth=self.bandwidth, max_iter=self.max_iter, bin_seeding=True)
                cluster_labels = ms.fit_predict(emb)  # [N]
                cluster_labels = torch.tensor(cluster_labels, device=embeddings.device)

                # assign cluster majority class
                seg_pred = torch.argmax(torch.tensor(seg, device=embeddings.device), dim=0)  # [N]
                refined = torch.zeros_like(seg_pred)
                for c in cluster_labels.unique():
                    mask = (cluster_labels == c)
                    if mask.sum() == 0:
                        continue
                    majority_class = seg_pred[mask].mode()[0]
                    refined[mask] = majority_class

                refined = nn.functional.one_hot(refined, num_classes=seg_logits.size(1))
                refined = refined.permute(1, 0).view_as(seg_logits[b])  # [n_classes, D, H, W]
            except Exception:
                # fallback if clustering fails
                refined = torch.softmax(seg_logits[b], dim=0)

            refined_segs.append(refined)

        refined_segs = torch.stack(refined_segs, dim=0)  # [B, n_classes, D, H, W]
        return refined_segs
