#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

# valid = (ground_truth > 0.1) & (ground_truth < 20)
# loss = torch.mean(torch.abs(absolute_depth_map[valid] - ground_truth[valid]) / ground_truth[valid]) 
        
# small_constant = 1e-6
small_constant = 0
class ValidatedLoss(nn.Module):
    """ValidatedLoss, """
    def __init__(self, basic_loss:nn.Module, lower=0.1, upper=20):
        super().__init__()
        self.basic_loss = basic_loss
        self.lower = lower
        self.upper = upper

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """input 是 y_pred, target 是 y_tru
        """
        is_valid = (target > self.lower) & (target < self.upper)
        valid_input = input[is_valid]
        valid_target = target[is_valid]
        return self.basic_loss(valid_input, valid_target)
class REL(nn.Module):
    """Absolute Relative Error, REL Loss"""
    def __init__(self):
        super(REL, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """input 是 y_pred, target 是 y_true
        """
        # return F.l1_loss(input, target, reduction=self.reduction)
        # 比如说 shape = (b, 1, w, h)
        # REL = (1/b) (1/(wh)) ∑|d_gt - d_pred|/d_gt
        # 同时在batch和空间上求平均，mean直接得到一个数。
        return torch.mean(torch.abs(input - (target+small_constant)) / (target+small_constant))

# RMSE 不是 MSE！
# l = loss(f(x, w), y), 求 dl/dw
# 1. 根号的导数不是导数本身，虽然单调性不变，但是数值不同。
# d sqrt(l)/dw = 1/2 * 1/sqrt(l) * dl/dw
# 2. 根号的平均值不是平均值的根号
# mean(sqrt(L)) != sqrt(mean(L))
# 因此 这篇博客的写法错误 https://blog.csdn.net/GENGXINGGUANG/article/details/127007337

class RMSE(nn.Module):
    """Root-Mean-Square Error, RMSE"""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        shape = input.shape
        dim_except_batch = list(range(1, len(shape)))
        return torch.mean(
            torch.sqrt(
                torch.mean(
                    torch.pow(input - target, 2), 
                    dim=dim_except_batch
                )
            )
        )
# class SI_RMSE(nn.Module):
#     """Scale Invariant Root Mean Squared Error, si-RMSE"""
#     def __init__(self):
#         super().__init__()

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         shape = input.shape
#         dim_except_batch = list(range(1, len(shape)))
#         log_input = torch.log(input+small_constant)
#         log_target = torch.log(target+small_constant)
#         return torch.mean(
#             torch.sqrt(
#                 torch.mean(
#                     torch.pow(
#                         log_input - log_target, 2), 
#                     dim=dim_except_batch
#                 )-torch.pow(
#                     torch.mean(
#                         log_input - log_target, 
#                         dim=dim_except_batch
#                     ), 2
#                     )
#                 )
#             )
#         # 定义 si rmse 作为损失函数
class SI_RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        # 取对数，避免负值或零值
        pred = torch.log(torch.clamp(pred, min=1e-6))
        gt = torch.log(torch.clamp(gt, min=1e-6))
        # 计算均方根误差
        rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
        # 计算对数深度值的均值
        mean = torch.mean(pred - gt)
        # 计算 si rmse
        si_rmse = torch.sqrt(torch.mean((rmse - mean) ** 2))
        return si_rmse

class LOG10(nn.Module):
    """Average log10 errors, LOG10"""
    # TODO
    pass

class CompetitionLoss(nn.Module):
    """Some Information about CompetitionLoss"""
    def __init__(self):
        super().__init__()
        self.rel = REL()
        self.si_rmse = SI_RMSE()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # scale = 1000
        scale = 1
        return 0.5*self.rel(input*scale, target*scale) + 0.5*self.si_rmse(input*scale, target*scale)


#%%
# 测试
if __name__ == "__main__":
    input = torch.ones(4, 1, 512, 512, requires_grad=True)
    # target = torch.zeros(4, 1, 512, 512)
    target = torch.rand(4, 1, 512, 512)
    target[0] = 0
    criteria = CompetitionLoss()
    # criteria = REL()
    loss = criteria(input, target) 
    loss.backward()
    print(loss)
#%%