import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def soft_process(loss):
    soft_loss = torch.log(1+loss+loss*loss/2)
    return soft_loss

def PLC_uncertain_discard(user, item, train_mat, y, t, drop_rate, epoch, sn, before_loss, co_lambda, relabel_ratio):
    # This is the core of our DCF
    before_loss = torch.from_numpy(before_loss).cuda().float().squeeze()
    
    s = torch.tensor(epoch + 1).float() # as the epoch starts from 0
    co_lambda = torch.tensor(co_lambda).float()    
    # 计算每个样本的二元交叉熵（带logits输入），不做reduce得到每个样本的损失，此处为论文所使用的损失函数，对应论文第二章BCE的公式``  
    # y对应模型预测，t对应真实标签，reduce=False表示不进行求和或平均，返回每个样本的损失值``  
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)
    
    # 只对正样本（t==1）关注损失，把负样本的损失置零（loss * t）
    loss_mul = loss * t
    # 对正样本损失做平滑处理（paper 中提出的 soft process）
    loss_mul = soft_process(loss_mul)  # soft process is non-decreasing damping function in the paper
    # 用之前的before_loss与当前loss_mul做指数/简单平均，计算历史平均损失，对应论文中第三章的公式（2）
    loss_mean = (before_loss * s + loss_mul) / (s + 1)   # computing mean loss in Eq.2.2
    
    # 计算置信界（confidence bound），对应论文中第三章的公式（3）
    confidence_bound = co_lambda * (s + (co_lambda * torch.log(2 * s)) / (s * s)) / ((sn + 1) - co_lambda)
    confidence_bound = confidence_bound.squeeze()
    
    # 只保留大于置信界的部分，作为高损失的判定依据，对应论文中第三章的公式（4）
    loss_mul = F.relu(loss_mean.float() - confidence_bound.cuda().float())  # loss low bound in Eq.4
    
    # The following is the code implementation of dropping and relabelling.
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]
    
    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))
    
    highest_ind_sorted = ind_sorted[int(((1-relabel_ratio)+relabel_ratio*remember_rate)*len(loss_sorted)):]
    # saved_ind_sorted是保留的索引（记住的前num_remember个）
    saved_ind_sorted = ind_sorted[:num_remember]   # 对应论文中的公式（6）
    final_ind = torch.concat((highest_ind_sorted, saved_ind_sorted))
    lowest_ind_sorted = ind_sorted[:int(((1-relabel_ratio)+relabel_ratio*remember_rate)*len(loss_sorted))]

    t = torch.tensor(train_mat[user.cpu().numpy().tolist(), item.cpu().numpy().tolist()].todense()).squeeze().cuda()
    loss_update = F.binary_cross_entropy_with_logits(y[final_ind], t[final_ind])
    
    return loss_update, train_mat, loss_mean, lowest_ind_sorted    


def loss_function(y, t, drop_rate):
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

    return loss_update






