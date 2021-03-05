import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from utilities import config
from model import attention
from train.loss_functions import *
import torch.nn.functional as F


def da_ass(pos_ans, gt_ans, k=5):
    pos_idx = pos_ans.topk(k, 1)[1].data  # argmax
    neg_ans = gt_ans.clone()
    neg_ans = neg_ans.scatter(1, pos_idx, 0)
    return neg_ans


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros_like(labels).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    indices = scores.long().sum(dim=1)
    return scores.sum(dim=1), (indices == 0)


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
    # print(hint_score[:3])
    return focus_false_objs, not_focus_objs, thresh


def instance_with_mask(logits, mask_logits, labels):
    assert logits.dim() == 2
    # pair_loss = PairWiseLoss(margin=config.margin)
    loss = binary_cross_entropy_with_logits(logits, labels)
    mask_loss = binary_cross_entropy_with_logits(mask_logits, labels)
    # score_loss = binary_cross_entropy_with_logits(mask_logits, torch.sigmoid(logits.data), True)
    # score_loss = F.kl_div(F.log_softmax(mask_logits), F.softmax(logits.data), reduction='batchmean')
    score_loss = js_div(mask_logits, logits.data)
    return loss.mean(), mask_loss.mean(), score_loss


def zero_grad(net, zero_mcls=True, zero_cls=True, zero_att=True):
    net.module.w_emb.zero_grad()
    net.module.q_emb.zero_grad()
    if zero_att:
        net.module.v_att.zero_grad()
    if zero_cls:
        net.module.classifier.zero_grad()
    if zero_mcls:
        net.module.mask_classifier.zero_grad()
    net.module.debias_loss_fn.zero_grad()


def run(model, loader, optimizer, tracker, train=False, has_answers=True, prefix='', epoch=0, args=None):
    if train:
        model.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        model.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ, mask_answ, q_ids, accs, weights, spatials, hints = [], [], [], [], [], [], []
    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    mask_acc_tracker = tracker.track('{}_m-acc'.format(prefix), tracker_class(**tracker_params))
    nf_ratio_tracker = tracker.track('{}_nf-ratio'.format(prefix), tracker_class(**tracker_params))
    eps = 1e-7
    for i, (v, b, q, a, qid, bias, hint_score, has_hint) in enumerate(loader):
        v = v.cuda().float().requires_grad_()
        b = b.cuda()
        q = q.cuda()
        a = a.cuda()
        hint_score = hint_score.cuda()
        has_hint = has_hint.cuda()
        neg_gta = torch.zeros_like(a)
        pred, loss, att = model(v, b, q, a, bias)
        if has_answers:
            acc, indices = compute_score_with_logits(pred.data, a.data)

        if model.training == False:
            _, mask_not_focus_objs, thresh = get_mask_btw_mfs_att(hint_score, att)  # [batch , num_objs]
            nf_ratio = (mask_not_focus_objs * has_hint.unsqueeze(-1)).sum() / (has_hint.sum() + eps)
            opt_mask = has_hint * (mask_not_focus_objs.sum(dim=1) > 0).float()
            mask_pred, _, _ = model(v * (~mask_not_focus_objs).unsqueeze(-1).float(), b, q, neg_gta, bias,
                                    has_hint=opt_mask)
            if has_answers:
                mask_acc, _ = compute_score_with_logits(mask_pred, a.data)
                mask_acc = (mask_acc * opt_mask).sum() / (opt_mask.sum() + eps)

        if train:
            if not (config.use_debias or config.use_rubi):
                loss = binary_cross_entropy_with_logits(pred, a, mean=True)
            (loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer['all_optimizer'].step()
            optimizer['all_optimizer'].zero_grad()
        else:
            # store information about evaluation of this minibatch
            answer_idx = pred.max(dim=1)[1].cpu()
            answ.append(answer_idx.view(-1))
            answer_idx = mask_pred.max(dim=1)[1].cpu()
            mask_answ.append(answer_idx.view(-1))
            q_ids.append(qid.view(-1))
            if has_answers:
                accs.append(acc.view(-1).cpu())
            weights.extend(att.detach().cpu().numpy())
            spatials.extend(b.detach().cpu().numpy())
            # indices = indices * has_hint.bool()
            # answ.append(answer_idx.view(-1)[indices])
            # q_ids.append(qid.view(-1)[indices])
            # weights.extend(att[indices].detach().cpu().numpy())
            # spatials.extend(b[indices].detach().cpu().numpy())
            # hints.extend(hint_score[indices].detach().cpu().numpy())
        if has_answers:
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            if train:
                loader.set_postfix(loss=fmt(loss_tracker.mean.value),
                                   acc=fmt(acc_tracker.mean.value),
                                   )
            else:
                mask_acc_tracker.append(mask_acc.mean())
                nf_ratio_tracker.append(nf_ratio.item())
                loader.set_postfix(loss=fmt(loss_tracker.mean.value),
                                   acc=fmt(acc_tracker.mean.value),
                                   m_acc=fmt(mask_acc_tracker.mean.value),
                                   nf_r=fmt(nf_ratio_tracker.mean.value),
                                   )
    if not train:
        answ = torch.cat(answ, dim=0).numpy()
        mask_answ = torch.cat(mask_answ, dim=0).numpy()
        q_ids = torch.cat(q_ids, dim=0).numpy()
        if has_answers:
            accs = torch.cat(accs, dim=0).cpu().numpy()
        return answ, accs, q_ids, weights, spatials, mask_answ, hints
