# -*- coding: utf-8 -*-
# @Time    : 2024/12/19 13:18
# @Author  : Biao
# @File    : loss.py


import torch
import torch.nn.functional as F
from torch import nn


class LogMSELoss(nn.Module):
    def __init__(self, epsilon=1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        def log_transform(x):
            return torch.sign(x) * torch.log(torch.abs(x) + self.epsilon)

        y_pred_log = log_transform(y_pred)
        y_true_log = log_transform(y_true)

        return F.mse_loss(y_pred_log, y_true_log)

variances = [4.31894849e+04, 6.14759054e+06, 1.30248315e+02, 6.11561697e+06,
            5.34715550e+01, 4.46133841e+10, 4.42557782e+10, 2.27609201e+06,
            6.18420611e+10, 3.65251049e+01, 1.17722669e+02, 2.36032998e+04,
            6.38952032e-11, 2.38130761e-11, 2.35571270e-11, 7.16103929e-08,
            7.12098610e-08, 1.03705739e-11, 1.11759655e+08, 1.28603442e+09,
            3.71531249e+00, 3.67222410e-05, 1.12620648e-07, 1.38980112e+03,
            9.41802433e+05]

class MultiTaskLossWrapper(nn.Module):
    """
    多任务损失函数包装器, 对每个任务的预测值和真实值进行对数变换后计算MSE损失(logMSE)
    """
    def __init__(self, model, num_tasks, epsilon=1,task_variances=variances):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        # self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.epsilon = epsilon
        if task_variances is not None:
            # 将方差转换到对数空间
            variances = torch.tensor(task_variances)
            # 对方差取对数，并进行归一化
            log_vars = torch.log(variances + self.epsilon)
            # 归一化到合理范围，例如[-1, 1]
            log_vars_normalized = 2 * (log_vars - log_vars.min()) / (log_vars.max() - log_vars.min()) - 1
            # 可以根据需要调整缩放因子
            scaling_factor = 0.5
            initial_log_vars = scaling_factor * log_vars_normalized
            self.log_vars = nn.Parameter(initial_log_vars)
        else:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, y_preds, y_true):
        losses = []
        num_predictions = min(len(y_preds), len(self.log_vars))
        for i in range(num_predictions):
            precision = torch.exp(-self.log_vars[i])
            y_pred_log = torch.sign(y_preds[i]) * torch.log(torch.abs(y_preds[i]) + self.epsilon)
            y_true_log = torch.sign(y_true[i]) * torch.log(torch.abs(y_true[i]) + self.epsilon)
            loss = precision * (y_pred_log - y_true_log) ** 2 + self.log_vars[i]
            losses.append(torch.mean(loss))
        total_loss = sum(losses)
        return total_loss

LOSS_FUNCS = {
    "LogMSE": LogMSELoss,
    "MSE": nn.MSELoss,
    "MultiTask": MultiTaskLossWrapper,
}
