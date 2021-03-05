import model.attention as attention
from model.language_model import WordEmbedding, QuestionEmbedding
from model.classifier import SimpleClassifier
from utilities import config
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from model.vqa_debias_loss_fuctions import *
from model.fc import  MLP, FCNet

# def bce_loss(input, target, mean=True):
#     """
#     Function that measures Binary Cross Entropy between target and output logits:
#     """
#     if not target.is_same_size(input):
#         raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
#     max_val = (-input).clamp(min=0)
#     loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
#     loss = loss.sum(dim=1)
#     return loss.mean() if mean else loss


class BaseModel_with_Onestep(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier, debias_loss_fn ,extra_c1, extra_c2):
        super(BaseModel_with_Onestep, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier
        self.debias_loss_fn = debias_loss_fn
        self.extra_c1 = extra_c1
        self.extra_c2 = extra_c2

    def forward(self, v, b, q, labels, bias, hint=None, has_hint=None):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        *_v_emb: [batch, g*v_dim], mask_weight: [batch, g]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        v_emb, v_att, mask_v_emb = self.v_att(v, q_emb, hint)  # [batch, v_dim]
        if config.att_norm:
            v_emb = attention.apply_norm_attention(v, v_att, mode='rand')
        joint_repr, logits = self.classifier(q_emb, v_emb)
        debias_loss = torch.zeros(1)
        if labels is not None:
            if config.use_debias:
                debias_loss = self.debias_loss_fn(joint_repr, logits, bias, labels, has_hint)
            elif config.use_rubi:
                q_pred = self.extra_c1(q_emb.detach())
                q_out = self.extra_c2(q_pred)
                rubi_logits = logits*torch.sigmoid(q_pred)
                if has_hint is not None:
                    debias_loss = bce_loss(rubi_logits, labels, False) + bce_loss(q_out, labels, False)
                    debias_loss = (debias_loss * has_hint).sum()/ has_hint.sum()
                else:
                    debias_loss = bce_loss(rubi_logits, labels) + bce_loss(q_out, labels)
                    debias_loss *= labels.size(1)

        return logits, debias_loss, v_att


class BaseModel_with_Twostep(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier, debias_loss_fn ,extra_c1, extra_c2):
        super(BaseModel_with_Twostep, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier
        self.debias_loss_fn = debias_loss_fn
        self.extra_c1 = extra_c1
        self.extra_c2 = extra_c2

    def forward(self, v, b, q, labels, bias, hint=None, has_hint=None):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        *_v_emb: [batch, g*v_dim], mask_weight: [batch, g]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        v_emb, v_att = self.v_att(v, q_emb, hint)  # [batch, v_dim]
        if config.att_norm:
            v_emb = attention.apply_norm_attention(v, v_att, mode='avg')
        joint_repr, logits = self.classifier(q_emb, v_emb)
        debias_loss = torch.zeros(1)
        if labels is not None:
            if config.use_debias:
                debias_loss = self.debias_loss_fn(joint_repr, logits, bias, labels, has_hint)
            elif config.use_rubi:
                q_pred = self.extra_c1(q_emb.detach())
                q_out = self.extra_c2(q_pred)
                rubi_logits = logits*torch.sigmoid(q_pred)
                if has_hint is not None:
                    debias_loss = bce_loss(rubi_logits, labels, reduction='none') + bce_loss(q_out, labels, reduction='none')
                    debias_loss = (debias_loss.sum(dim=1) * has_hint).sum()/ has_hint.sum()
                else:
                    debias_loss = bce_loss(rubi_logits, labels) + bce_loss(q_out, labels)
                    debias_loss *= labels.size(1)

        return logits, debias_loss, v_att


def build_baseline_with_onestep(embeddings, num_ans_candidates, debias_mode='LearnedMixin'):
    assert debias_mode in ['BiasProduct', 'ReweightByInvBias', 'LearnedMixin', 'Plain']
    vision_features = config.output_features
    visual_glimpses = config.visual_glimpses
    hidden_features = config.hid_dim
    question_features = config.hid_dim
    w_emb = WordEmbedding(
        embeddings,
        dropout=0.0
    )
    q_emb = QuestionEmbedding(
        w_dim=300,
        hid_dim=question_features,
        nlayers=1,
        bidirect=False,
        dropout=0.0
    )

    v_att = attention.Attention(
        v_dim=vision_features,
        q_dim=question_features,
        hid_dim=hidden_features,
        glimpses=visual_glimpses,
    )

    classifier = SimpleClassifier(
        in_dim=(question_features, visual_glimpses * vision_features),
        hid_dim=(hidden_features, hidden_features * 2),
        out_dim=num_ans_candidates,
        dropout=0.5
    )

    # mask_v_att = attention.Attention(
    #     v_dim=vision_features,
    #     q_dim=question_features,
    #     hid_dim=hidden_features,
    #     glimpses=visual_glimpses,
    # )
    #
    # mask_classifier = SimpleClassifier(
    #     in_dim=(question_features, vision_features),
    #     hid_dim=(hidden_features, hidden_features * 2),
    #     out_dim=num_ans_candidates,
    #     dropout=0.5
    # )
    # Add the loss_fn based our arguments
    debias_loss_fn = eval(debias_mode)()
    return BaseModel_with_Onestep(w_emb, q_emb, v_att, classifier, debias_loss_fn)


def build_baseline_with_twostep(embeddings, num_ans_candidates, debias_mode='LearnedMixin'):
    assert debias_mode in ['BiasProduct', 'ReweightByInvBias', 'LearnedMixin', 'Plain']
    vision_features = config.output_features
    visual_glimpses = config.visual_glimpses
    hidden_features = config.hid_dim
    question_features = config.hid_dim
    w_emb = WordEmbedding(
        embeddings,
        dropout=0.0
    )
    q_emb = QuestionEmbedding(
        w_dim=300,
        hid_dim=question_features,
        nlayers=1,
        bidirect=False,
        dropout=0.0
    )

    v_att = attention.Attention(
        v_dim=vision_features,
        q_dim=question_features,
        hid_dim=hidden_features,
        glimpses=visual_glimpses,
    )

    classifier = SimpleClassifier(
        in_dim=(question_features, visual_glimpses * vision_features),
        hid_dim=(hidden_features, hidden_features * 2),
        out_dim=num_ans_candidates,
        dropout=0.5
    )

    if config.use_rubi:
        c1 = MLP(
            input_dim=question_features,
            dimensions=[1024, 1024, num_ans_candidates],
        )
        c2 = nn.Linear(num_ans_candidates, num_ans_candidates)
    else:
        c1, c2 = None, None

    # Add the loss_fn based our arguments
    debias_loss_fn = eval(debias_mode)(hidden_features if config.fusion_type=='mul' else hidden_features*2)
    return BaseModel_with_Twostep(w_emb, q_emb, v_att, classifier, debias_loss_fn, c1, c2)