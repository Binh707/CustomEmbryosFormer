import torch
from sklearn.metrics import precision_score

__all__ = ['Precision']


class Precision():
    def __init__(self, nclasses, ignore_classes=None, mode='macro'):
        self.mode = mode
        self.labels = list(range(nclasses))
        if ignore_classes is not None:
            self.labels = list(
                filter(lambda x: x not in ignore_classes,
                       self.labels)
            )
        self.reset()

    def update(self, output, target, is_prob=True):
        if is_prob:
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
        self.pred += pred.cpu().tolist()
        self.target += target.cpu().tolist()

    def reset(self):
        self.pred = []
        self.target = []

    def value(self):
        return precision_score(self.target, self.pred,
                        labels=self.labels, average=self.mode, zero_division=0)

    def report(self):
        report_str = ''
        f1 = precision_score(self.target, self.pred,
                      labels=self.labels, average=None, zero_division=0)
        for c, s in zip(self.labels, f1):
            report_str += f'\t{c}: {s}'
        
        return report_str


    def summary(self, logger=None):
        func = logger.info if logger is not None else print
        func(f'+ Precision:')

        for mode in ['micro', 'macro', 'weighted']:
            f1 = precision_score(self.target, self.pred,
                          labels=self.labels, average=mode, zero_division=0)
            func(f'{mode}: {f1}')

        func(f'class:')
        f1 = precision_score(self.target, self.pred,
                      labels=self.labels, average=None, zero_division=0)
        for c, s in zip(self.labels, f1):
            func(f'\t{c}: {s}')
