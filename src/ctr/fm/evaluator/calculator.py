import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryPrecision, 
    BinaryRecall,
    BinaryF1Score, 
    BinaryConfusionMatrix,
    BinaryAUROC,
)


class Calculator(object):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.accuracy = BinaryAccuracy(threshold=threshold)
        self.precision = BinaryPrecision(threshold=threshold)
        self.recall = BinaryRecall(threshold=threshold)
        self.f1 = BinaryF1Score(threshold=threshold)
        self.confmat = BinaryConfusionMatrix(threshold=threshold)
        self.auroc = BinaryAUROC()

    def __call__(self, result):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.auroc.reset()

        args = (result["prob"], result["true"])
        
        self._confmat(*args)

        data = dict(
            threshold=self.threshold,
            accuracy=self.accuracy(*args).item(),
            precision=self.precision(*args).item(),
            recall=self.recall(*args).item(),
            f1=self.f1(*args).item(),
            auroc=self.auroc(*args).item(),
            logloss=F.binary_cross_entropy(*args).item(),
        )
        return pd.DataFrame([data])

    def _confmat(self, *args):
        self.confmat.reset()

        kwargs = dict(
            data=self.confmat(*args).cpu().numpy(), 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=False,
            xticklabels=["NONCLICK", "CLICK"],
            yticklabels=["NONCLICK", "CLICK"],
        )

        plt.figure(figsize=(5,5))
        sns.heatmap(**kwargs)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Binary Confusion Matrix")
        plt.tight_layout()
        plt.show()
