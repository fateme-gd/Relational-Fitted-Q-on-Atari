import abc


class LearningRateStrategy(object, metaclass=abc.ABCMeta):

    def reset(self):
        pass

    def end_epoch(self):
        pass
