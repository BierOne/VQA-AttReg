import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from model.fc import FCNet
import utilities.config as config


def convert_tensor_to_binary_with_topk(input, position):
    # get top-m input with binary form
    input_sort, input_idx = input.sort(1, descending=True)
    thresh = input_sort[:, position - 1:position] - 1e-5
    # thresh += ((thresh < 0.2).float() * 0.1)
    return (input > thresh), thresh


def get_mask_btw_mfs_att(hint_score, att_weights):
    att_weights = att_weights[..., 0]  # Remove the objects follow the specified att-layer
    att_weights, thresh = convert_tensor_to_binary_with_topk(att_weights, config.masks)
    focus_false_objs = (~hint_score) & att_weights
    not_focus_objs = hint_score & (~att_weights)
    return focus_false_objs, not_focus_objs, thresh


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, glimpses=1, dropout=0.2):
        super(Attention, self).__init__()

        self.v_proj = FCNet([v_dim, hid_dim], dropout)
        self.q_proj = FCNet([q_dim, hid_dim], dropout)
        self.drop = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hid_dim, glimpses), dim=None)

    def forward(self, v, q, hint_score=None):
        """
        v: [b, o, vdim]
        q: [b, qdim]
        """
        v_proj = self.v_proj(v)  # [b, o, hid_dim]
        q_proj = self.q_proj(q).unsqueeze(1) # [b, 1, hid_dim]
        att_logits = self.linear(self.drop(v_proj * q_proj)) # [b, o, g]
        v_emb, v_att = apply_attention(v, att_logits) # [b, o, g]
        return v_emb, v_att
        # _, mask_v_emb, topm_idx, rest_idx = apply_mask_to_visual_features(v, att_logits)
        # mask_v_emb, _ = apply_hardmask_with_logits(v, att_logits)
        # mask_val = v_att.gather(dim=1, index=topm_idx).sum(dim=1) # [b, m, g] -> [b, g]
        # if hint_score is not None:
        #     mask_v_emb, _ = apply_mask_by_hint(v, att_logits, hint_score)
        # else:
        #     mask_v_emb = None
        # return v_emb, v_att, mask_v_emb


def apply_mask_by_hint(input, att_logits, hint):
    _, mask_not_focus_objs, _ = get_mask_btw_mfs_att(hint, att_logits) # [batch , num_objs], positive masks
    att = torch.exp(att_logits - torch.max(att_logits))
    att = att * (~mask_not_focus_objs).unsqueeze(-1).float()
    att = att / (att.sum(dim=1, keepdim=True) + 1e-6)

    input = input.unsqueeze(2) # [b, o, 1, v]
    weighted = att * input # [b, o, g, v]
    weighted_mean = weighted.sum(dim=1) # [b, g, v]
    return weighted_mean


def set_mask_by_gradcam(input, v_grad_logits):
    v_grad_logits = v_grad_logits.unsqueeze(-1)  # [b, o, 1]
    v_att = F.softmax(v_grad_logits, dim=1)
    _, mask_v_emb,  topm_idx, rest_idx = apply_mask_to_visual_features(input, v_grad_logits)
    mask_val = v_att.gather(dim=1, index=topm_idx).sum(dim=1)  # [b, m, 1] -> [b, 1]
    return v_att, mask_v_emb, mask_val.mean()


def apply_attention(input, att_logits):
    """Apply any number of attention maps over the input.

    input: [batch, objects, visual_features]
    att_logits: [batch, objects, glimpses]
    """
    b, o, v = input.shape # [b, o, v]
    attention = F.softmax(att_logits, dim=1).unsqueeze(-1) # [b, o, g, 1]
    input = input.unsqueeze(2) # [b, o, 1, v]
    weighted = attention * input # [b, o, g, v]
    weighted_mean = weighted.sum(dim=1) # [b, g, v]
    return weighted_mean.view(b, -1), attention.squeeze(-1)


def apply_norm_attention(input, att_logits, mode='avg'):
    """Apply norm functions over the input.

    input: [batch, objects, visual_features]
    att_logits: [batch, objects, glimpses]
    """
    b, o, v = input.shape # [b, o, v]
    if mode == 'avg':
        norm = torch.ones_like(att_logits).cuda() * (1.0 / o)  # average
    elif mode == 'rand':
        norm = F.softmax(torch.randn_like(att_logits).cuda(), dim=-1)  # random
    else:
        norm = torch.zeros_like(att_logits).cuda()  # zeros
    norm = norm.unsqueeze(-1)  # [b, o, g, 1]
    input = input.unsqueeze(2)  # [b, o, 1, v]
    weighted = norm * input # [b, o, g, v]
    weighted_mean = weighted.sum(dim=1) # [b, g, v]
    return weighted_mean.view(b, -1)


def apply_hardmask_with_logits(input, logits, mask=config.masks, mask_lower_w=False):
    """ Apply any number of attention masks over the input.

        input: [batch, num_objs, visual_features] = [b, o ,v]
        attention: [batch, num_objs, glimpses] = [b, o ,g]
        mask: number of masks
        mask_lower_w: if True, mask lower att weights;
                      if False, mask higher att weights
        return: attended_input, masked_weight
    """
    def _mask_softmax_with_logits(logits, mask, mask_lower_w):
        att = torch.exp(logits)  # [b, o, g]
        # _, mask_idx = att.topk(mask, dim=1, largest=not mask_lower_w)  # [b, m, g]
        mask_idx = att.sort(1, descending=not mask_lower_w)[1][:, :mask, :]
        att = att.scatter(1, mask_idx, 1e-6)
        att = att / att.sum(dim=1, keepdim=True)
        return att, mask_idx

    b, o, v = input.shape
    # remain the lower att-weights
    masked_att, mask_idx = _mask_softmax_with_logits(logits, mask, mask_lower_w) # [b, o, g], [b, m, g]
    input = input.unsqueeze(2)  # [b, o, 1, v]
    masked_att = masked_att.unsqueeze(-1)  # [b, o, g, 1]
    weighted = masked_att * input  # [b, o, g, v]
    weighted_mean = weighted.sum(dim=1) # [b, g, v]
    return weighted_mean.view(b, -1), mask_idx


def split_att_use_mask(logits, mask):
    """ split the objects with the top-k attention weights by using att-logits
    """
    o = logits.size(1)  # [b, o, 1]
    rank_idx = logits.topk(o, dim=1)[1]
    topm_idx = rank_idx[:, :mask, ...] # [b, m, 1]
    rest_idx = rank_idx[:, mask:, ...] # [b, o-m, 1]
    return topm_idx, rest_idx


def apply_mask_to_visual_features(input, att_logits, mask=config.masks, layer=0, mask_lower_w=False):
    """ Remove the objects follow the specified att-layer

        input: [batch, num_objs, visual_features] = [b, o ,v]
        att_logits: [batch, num_objs, glimpses] = [b, o ,g]
        mask: number of masks
        mask_lower_w: if True, mask the visual_features with lower att weights;
                 if False, mask the visual_features with higher att weights
        return: masked_input, attended_input
    """
    b, o, v = input.shape
    # Remove the objects follow the specified att-layer
    att_logits = att_logits[...,layer].unsqueeze(-1) # [b, o, 1]
    topm_idx, rest_idx = split_att_use_mask(att_logits, mask)
    if mask_lower_w:
        att_logits = att_logits.gather(dim=1, index=topm_idx)  # [b, m, 1]
        masked_input = input.gather(dim=1, index=topm_idx.expand(-1, -1, v))  # [b, m, v]
    else:
        att_logits = att_logits.gather(dim=1, index=rest_idx)  # [b, o-m, 1]
        masked_input = input.gather(dim=1, index=rest_idx.expand(-1, -1, v))  # [b, o-m, v]

    masked_att = F.softmax(att_logits, dim=1) # [b, m, 1]
    weighted = masked_att * masked_input  # [b, o-m, v] or [b, m, v]
    weighted_mean = weighted.sum(dim=1)  # [b, v]
    return masked_input, weighted_mean, topm_idx, rest_idx


def split_visual_features_by_vss(input, att_logits, mask=config.masks, layer=0):
    """ Remove the objects follow the specified att-layer

        input: [batch, num_objs, visual_features] = [b, o ,v]
        att_logits: [batch, num_objs, glimpses] = [b, o ,g]
        mask: number of masks
        mask_lower_w: if True, mask the visual_features with lower att weights;
                 if False, mask the visual_features with higher att weights
        return: masked_input, attended_input
    """
    def _remove_topk_obj_with_logits(logits, mask):
        """ Remove the objects with the top-k attention weights
        """
        o = logits.size(1)  # [b, o, 1]
        rank_idx = logits.topk(o, dim=1)[1]
        topm_idx = rank_idx[:, :mask, ...] # [b, m, 1]
        rest_idx = rank_idx[:, mask:, ...] # [b, o-m, 1]

        return topm_idx, rest_idx

    b, o, v = input.shape
    # Remove the objects follow the specified att-layer
    att_logits = att_logits[...,layer].unsqueeze(-1) # [b, o, 1]
    topm_idx, rest_idx = _remove_topk_obj_with_logits(att_logits, mask)
    topk_input = input.gather(dim=1, index=topm_idx.expand(-1, -1, v))  # [b, m, v]
    rest_input = input.gather(dim=1, index=rest_idx.expand(-1, -1, v))  # [b, o-m, v]
    return topk_input, rest_input


def apply_hardmask_with_map(input, attention, mask):
    """ Apply any number of attention masks over the input.

        input: [batch, num_objs, visual_features] = [b, o ,v]
        attention: [batch, num_objs, glimpses] = [b, o ,g]
        return: masked_input, masked_weight
    """
    b, o, v = input.shape
    # remain the lower att-weights
    mask_map = torch.ones_like(attention)  # [b, o, g]
    mask_val, mask_idx = attention.topk(mask, dim=1)  # [b, m, g]
    mask_map = mask_map.scatter(1, mask_idx, 0.0)  # [b, o, g]
    return input * mask_map
    # attention = attention * mask_map  # [b, o, g]
    # input = input.unsqueeze(2)  # [b, o, 1, v]
    # attention = attention.unsqueeze(-1)  # [b, o, g, 1]
    # weighted = attention * input  # [b, o, g, v]
    # weighted_mean = weighted.sum(dim=1)  # [b, g, v]
    # return weighted_mean.view(b, -1), mask_idx


def apply_softmask_with_att(input, attention):
    def _relu(x, c):
        return torch.clamp(x, max=c)

    def _sigmoid(x, w=50, c=0.1):
        c = c * torch.ones_like(x).cuda()
        return 1 / (1 + torch.exp(-w * (x - c)))

    b, o, g = attention.shape
    input = input.unsqueeze(2)  # [b, o, 1, v]
    attention = _sigmoid(attention)  # [b, o, g, 1]
    attention = attention.unsqueeze(-1)  # [b, o, g, 1]
    weighted = input - (attention * input)  # [b, o, g, v]
    weighted_mean = weighted.sum(dim=1)  # [b, g, v]
    return weighted_mean.view(b, -1)