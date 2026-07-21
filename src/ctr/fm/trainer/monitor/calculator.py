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

    def __call__(self, result):
        self.auroc.reset()
        self.roc.reset()

        args = (result["prob"], result["true"])

        return dict(
            score=self.auroc(*args).item(),
            threshold=self.threshold(*args).item(),
        )

    def threshold(self, *args):
        # Youden's J Statistic
        fpr, tpr, thresholds = self.roc(*args)
        return thresholds[torch.argmax(tpr - fpr)]