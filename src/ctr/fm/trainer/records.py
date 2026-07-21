from dataclasses import dataclass, field


@dataclass
class Records:
    trn_scores: list[float] = field(default_factory=list)
    val_scores: list[float] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)
    best_epoch: int = 0

    def update(self, state):
        self.trn_scores.append(state.trn_score)
        self.val_scores.append(state.val_score)
        self.thresholds.append(state.threshold)
        self.best_epoch = state.best_epoch

    def get(self):
        return dict(
            trn=self.trn_scores,
            val=self.val_scores,
            threshold=self.thresholds[self.best_epoch-1],
        )