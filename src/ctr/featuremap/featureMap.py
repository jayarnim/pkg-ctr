from .featureInfo import FeatureInfo


class FeatureMap(object):
    def __init__(
        self, 
        features: list[FeatureInfo],
    ):
        super().__init__()

        self.features = features

        self.by_name = {
            feature.name: feature
            for feature in features
        }

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return iter(self.features)

    def __getitem__(self, name):
        return self.by_name[name]