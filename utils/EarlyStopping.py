#
# 您可以把这个类放在一个单独的文件里，比如 utils/early_stopping.py
#
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            mode (str): 'min' or 'max'. In 'min' mode, training will stop when the quantity 
                        monitored has stopped decreasing; in 'max' mode it will stop when the 
                        quantity monitored has stopped increasing.
                        Default: 'min'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode

    def __call__(self, val_score, model, accelerator):
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, accelerator)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, accelerator)
            self.counter = 0

    def save_checkpoint(self, val_score, model, accelerator):
        '''Saves model when validation score improves.'''
        if self.verbose:
            if self.mode == 'min':
                self.trace_func(
                    f'Validation score decreased ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
            else:
                self.trace_func(
                    f'Validation score increased ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')

        # 使用 accelerator.unwrap_model 来获取原始模型
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), self.path)
        self.val_score_min = val_score