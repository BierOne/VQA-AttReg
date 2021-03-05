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
    return scores.sum(dim=1)


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

def split_with_mask_ratio(logits, mask_ratio):
    """ split the objects with the 10% attention weights by using att-logits
    """
    o = logits.size(1)  # [b, o, 1]
    mask_start = round(mask_ratio * o)
    rank_idx = logits.topk(o, dim=1)[1]
    topm_idx = rank_idx[:, mask_start:mask_start+4, ...] # [b, m, 1]
    return topm_idx


def run(model, loader, optimizer, tracker, train=False, has_answers=True, prefix='', epoch=0, mask_pos=0, args=None):
    if train:
        model.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        model.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ, mask_answ, q_ids, accs, accs_abs, weights, spatials, grad_weights = [], [], [], [], [], [], [], []
    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    mask_acc_tracker = tracker.track('{}_m-acc'.format(prefix), tracker_class(**tracker_params))
    mask_loss_tracker = tracker.track('{}_m-loss'.format(prefix), tracker_class(**tracker_params))
    score_loss_tracker = tracker.track('{}_s-loss'.format(prefix), tracker_class(**tracker_params))
    q_grad_tracker = tracker.track('{}_q-grad-weight'.format(prefix), tracker_class(**tracker_params))
    v_grad_tracker = tracker.track('{}_v-grad-weight'.format(prefix), tracker_class(**tracker_params))
    att_diff_tracker = tracker.track('{}_att-diff'.format(prefix), tracker_class(**tracker_params))
    sp_correlation_tracker = tracker.track('{}_sp'.format(prefix), tracker_class(**tracker_params))

    masks = torch.linspace(1, 35, 35).long().cuda()
    if mask_pos > 0:
        masks[mask_pos-1] = 0
    for i, (v, b, q, a, qid, bias, hint_score, has_hint) in enumerate(loader):
        v = v.cuda().float().requires_grad_()
        b = b.cuda()
        q = q.cuda()
        a = a.cuda()

        pred, loss, att = model(v, b, q, a, bias)
        if has_answers:
            acc = compute_score_with_logits(pred.data, a.data)
            # grad_att = torch.autograd.grad(pred.mean(dim=1).sum(), v, create_graph=True)[0]
            # grad_att = torch.autograd.grad((pred * (a > 0).float()).sum(), v, create_graph=True)[0]
            # grad_att = torch.autograd.grad((pred.max(dim=1)[0]).sum(), v, create_graph=True)[0]  # [batch , num_objs, obj_dim]
            grad_att = torch.autograd.grad((pred.topk(k=5, dim=1)[0]).sum(), v, create_graph=True)[0]
            grad_att = F.softmax(grad_att.sum(dim=2), dim=1).unsqueeze(-1) # [batch , num_objs, 1]
            topm_idx = split_with_mask_ratio(att if args.mask_type == 'att' else grad_att, mask_ratio=args.mask_ratio)

            # att_diff = mse(grad_att, att).mean()
            # sp_correlation = compute_rank_correlation(grad_att.data, att.data)

            mask_pred, mask_loss, _ = model(v.gather(dim=1, index=topm_idx.contiguous().expand(-1, -1, config.output_features)), b, q, a, bias)
            # mask_pred, mask_loss, _ = model(torch.zeros_like(v), b, q, a, bias)
            mask_acc = compute_score_with_logits(mask_pred.data, a.data)
        if train:
            # second step, use vqa(v-, a-) to update the parameters
            loss = binary_cross_entropy_with_logits(pred, a, mean=True)
            mask_loss = binary_cross_entropy_with_logits(mask_pred, a, mean=True)
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
            if has_answers:
                accs.append(acc.view(-1).cpu())
            q_ids.append(qid.view(-1))
            weights.extend(att.detach().cpu().numpy())
            grad_weights.extend(grad_att.detach().cpu().numpy())

        if has_answers:
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())

            # att_diff_tracker.append(att_diff.item())
            # sp_correlation_tracker.append(sp_correlation.item())
            mask_acc_tracker.append(mask_acc.mean())
            mask_loss_tracker.append(mask_loss.item())

            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value),
                               acc=fmt(acc_tracker.mean.value),
                               m_acc=fmt(mask_acc_tracker.mean.value),
                               mask_loss=fmt(mask_loss_tracker.mean.value),
                               # att_diff=fmt(att_diff_tracker.mean.value),
                               # sp=fmt(sp_correlation_tracker.mean.value),
                               )
    if not train:
        answ = torch.cat(answ, dim=0).numpy()
        mask_answ = torch.cat(mask_answ, dim=0).numpy()
        q_ids = torch.cat(q_ids, dim=0).numpy()
        if has_answers:
            accs = torch.cat(accs, dim=0).cpu().numpy()
        return answ, accs, q_ids, weights, spatials, mask_answ, grad_weights
