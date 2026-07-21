import torch
from torchmetrics.classification import (
    BinaryROC,
    BinaryAUROC,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Calculator(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.roc = BinaryROC().to(DEVICE)
        self.auroc = BinaryAUROC().to(DEVICE)

    def __call__(self, prob, true, state):
        # Youden's J Statistic
        self.roc.reset()
        fpr, tpr, thresholds = self.roc(prob, true)
        state.threshold = thresholds[torch.argmax(tpr - fpr)].item()
        
        # AUROC
        self.auroc.reset()
        state.val_score = self.auroc(prob, true).item()