import torch
import torch.nn as nn
from utilities import config
import torch.nn.functional as F


def compute_nll_loss_with_logits(input, traget):
    nll = -F.log_softmax(input, dim=1)
    loss = (nll * traget).sum(dim=1)
    return loss

def compute_rank_correlation(att, grad_att):
    """
    Function that measures Spearman’s correlation coefficient between target and output logits:
    """
    def _rank_correlation_(att_map, att_gd):
        n = torch.tensor(config.rcnn_output_size)
        upper = 6 * torch.sum((att_gd - att_map).pow(2), dim=1) # [batch, g]
        down = n * (n.pow(2) - 1.0)
        return (1.0 - (upper / down)).mean(dim=-1)

    att = att.sort(dim=1)[1] # [batch , num_objs, g]
    grad_att = grad_att.sort(dim=1)[1] # [batch , num_objs, 1]
    correlation = _rank_correlation_(att.float(), grad_att.float())
    return correlation.mean()


def binary_cross_entropy_with_logits(input, target, mean=False):
    """
    Function that measures Binary Cross Entropy between target and output logits:
    """
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    loss = loss.sum(dim=1)
    return loss.mean() if mean else loss


class PairWiseLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(PairWiseLoss, self).__init__()
        self.loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, x1, x2):
        y = torch.ones(x1.shape[-1]).cuda()
        return self.loss(x1, x2, y)


# def mse(mask_output, output, get_softmax=True):
#     """
#     Function that measures mse-loss divergence between target and output logits:
#     """
#     if get_softmax:
#         mask_output = F.softmax(mask_output, dim=-1)
#         output = F.softmax(output, dim=-1)
#     return (mask_output-output).abs().mean()


def mse(mask_output, output, get_sigmoid=False):
    """
    Function that measures mse-loss divergence between target and output logits:
    """
    if get_sigmoid:
        mask_output = torch.sigmoid(mask_output)
        output = torch.sigmoid(output)
    return (mask_output - output).pow(2).sum(dim=1)


def kld(input, target):
    log_cross_item = (target/input).log()
    return torch.sum(target* log_cross_item, dim=-1).mean()


def js_div(input, target, get_sigmoid=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    if get_sigmoid:
        input = F.sigmoid(input)
        target = F.sigmoid(target)
    input = torch.stack([input, 1-input], 2)
    target = torch.stack([target, 1-target], 2)
    mean_output = ((target + input)/2)
    return (kld(mean_output, input) + kld(mean_output, target))/2