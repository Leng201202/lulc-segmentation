import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.label_maps import IGNORE_INDEX


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX, epsilon: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, -50.0, 50.0)
        probs = F.softmax(logits, dim=1)
        valid = targets != self.ignore_index

        dice_scores = []
        for cls in range(self.num_classes):
            pred = probs[:, cls]
            target = (targets == cls).float()
            mask = valid.float()
            
            intersection = (pred * target * mask).sum()
            union = (pred * mask).sum() + (target * mask).sum()
            
            if union > self.epsilon:
                dice_score = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1.0 - dice_score
                # Clamp to prevent extreme values
                dice_loss = torch.clamp(dice_loss, 0.0, 1.0)
                dice_scores.append(dice_loss)

        if not dice_scores:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(dice_scores).mean()


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
        aux_weight: float = 0.4,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, tuple):
            main_logits, aux_logits = outputs
            main_loss = self.ce(main_logits, targets) + self.dice(main_logits, targets)
            aux_loss = self.ce(aux_logits, targets) + self.dice(aux_logits, targets)
            return main_loss + self.aux_weight * aux_loss

        return self.ce(outputs, targets) + self.dice(outputs, targets)
