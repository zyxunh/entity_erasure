from unhcv.common.utils import MeanCache


__all__ = ["BaseMetric"]


class BaseMetric:
    mean_cache = MeanCache()

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred, prefix=None):
        raise NotImplementedError

    def reset(self):
        self.mean_cache.clear()

    def update(self, y_true, y_pred):
        raise NotImplementedError

    def result(self):
        return self.mean_cache.mean(clear=True)

    def __str__(self):
        return self.__class__.__name__