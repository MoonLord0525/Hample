import numpy as np
import torch


class EarlyStopping:
    """
        Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float):  Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
            定义 __call__ 函数 -> 将一个类视作一个函数
            该函数的目的 类似在class中重载()运算符
            使得这个类的实例对象可以和普通函数一样 call
            即，通过 对象名() 的形式使用
        """

        score = -val_loss

        if self.best_score is None:
            """
                初始化（第一次call EarlyStopping）
                保存检查点
            """
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            """
                验证集损失没有继续下降时，计数
                当计数 大于 耐心值时，停止
                注：
                    由于模型性能没有改善，此时是不保存检查点的
            """
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            """
                验证集损失下降了，此时从头开始计数
                保存检查点
            """
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
            Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        """
            保存最优的模型Parameters
        """
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss