import sys, os
import spacy
import pickle, h5py, json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import argparse
sys.path.append(os.getcwd())
from utilities.dataset import Dictionary
from utilities import config


def filter_object_attributes(nlp, exp):
    doc = nlp(exp)
    object_attributes = []
    for token in doc:
        if token.pos_ == u'NUM' and token.dep_ == u'nummod' and token.head.pos_ == u'NOUN':
            obj = token.head.text
            attri = token.text
            key = str(obj) + ';' + str(attri)
            object_attributes.append(key)
        elif token.pos_ == u'ADJ' and token.dep_ == u'amod' and token.head.pos_ == u'NOUN':
            obj = token.head.text
            attri = token.text
            key = str(obj) + ';' + str(attri)
            object_attributes.append(key)
        elif token.pos_ == u'NOUN' and token.dep_ == u'compound' and token.head.pos_ == u'NOUN':
            obj = token.head.text
            attri = token.text
            key = str(obj) + ';' + str(attri)
            object_attributes.append(key)
    return object_attributes


def filter_objects(nlp, exp):
    obj_tokens = ""
    for k in ['question', 'answer']:
        doc = nlp(exp[k])
        for token in doc:
            if token.pos_ == u'NOUN':
                obj_tokens += " " + token.text
        print(exp[k])
    print(obj_tokens)
    return obj_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='vqx', help='explanation from vqx-exp or vqa-qa')
    args = parser.parse_args()
    return args


def main():
    assert config.rcnn_output_size == 36
    args = parse_args()
    qid2hint = {}
    with h5py.File(config.rcnn_trainval_path, 'r') as hf:
        img_id2idx = {img_id: idx for idx, img_id in enumerate(hf.get('ids'))}
        cls_scores = np.array(hf.get('cls_score'))
        attr_scores = np.array(hf.get('attr_score'))

    obj_emb_dict = np.load(os.path.join(config.main_path, 'glove6b_init_objects_300d.npy'))
    attr_emb_dict = np.load(os.path.join(config.main_path, 'glove6b_init_attributes_300d.npy'))
    if args.exp == 'vqx':
        exp_emb_dict = np.load(os.path.join(config.main_path, 'glove6b_init_vqx_300d.npy'))
        exp_dict = Dictionary.load_from_file(os.path.join(config.main_path, 'exp_dictionary.json'))
        qid2exp = pickle.load(open(os.path.join(config.main_path, 'vqx_qid2exp.pkl'), 'rb'))
    else:
        exp_emb_dict = np.load(os.path.join(config.cache_root, 'glove6b_init_vqa_caption_300d.npy'))
        exp_dict = Dictionary.load_from_file(os.path.join(config.cache_root, 'vqa_caption_dictionary.json'))
        qid2exp = pickle.load(open(os.path.join(config.cache_root, 'vqa_qid2exp.pkl'), 'rb'))
    qids = list(qid2exp.keys())
    nlp = spacy.load('en_core_web_lg')

    mod = 1000 if config.version=='v2' else 10
    for i in tqdm(range(len(qids))):
        qid = qids[i]
        img_id = qid // mod
        img_idx = img_id2idx[img_id]
        clss = cls_scores[img_idx][:, 0].astype('int')
        attrs = attr_scores[img_idx][:, 0].astype('int')

        objs = obj_emb_dict[clss, :]
        atts = attr_emb_dict[attrs, :]

        exp = qid2exp[qid]
        if args.exp == 'vqx':
            tokens = exp_dict.tokenize(exp, False)
            object_attributes = filter_object_attributes(nlp, exp)
        else:
            tokens = exp_dict.tokenize(exp['question'], False)
            tokens += exp_dict.tokenize(exp['answer'], False)
            object_attributes = filter_object_attributes(nlp, exp['question'])
            object_attributes += filter_object_attributes(nlp, exp['answer'])
            # tokens = exp_dict.tokenize(filter_objects(nlp, exp), False)

        exp_emb = exp_emb_dict[tokens]
        obj_sim = cosine_similarity(objs, exp_emb)

        hint_score = np.zeros((36))
        # hint_score_attr = np.zeros((36))

        # mentioned_tokens = []
        # mentioned_attrs = []

        for j in range(36):
            for k in range(len(tokens)):
                if obj_sim[j, k] > 0.6:
                    if hint_score[j] <= obj_sim[j, k]:
                        hint_score[j] = obj_sim[j, k]
            for key in object_attributes:
                obj, attr = key.split(';')
                # print(key, exp)
                obj_token = exp_dict.tokenize(obj, False)[0]
                attr_token = exp_dict.tokenize(attr, False)[0]
                if cosine_similarity(exp_emb_dict[obj_token:obj_token + 1], objs[j:j + 1]) > 0.3:
                    if cosine_similarity(exp_emb_dict[attr_token:attr_token + 1], atts[j:j + 1]) > 0.3:
                        hint_score[j] = cosine_similarity(exp_emb_dict[attr_token:attr_token + 1], atts[j:j + 1])

        qid2hint[qid] = hint_score
    if args.exp == 'vqx':
        pickle.dump(qid2hint, open(os.path.join(config.main_path, 'vqx_hint.pkl'), 'wb'))
    else:
        pickle.dump(qid2hint, open(os.path.join(config.cache_root, 'qa_hint.pkl'), 'wb'))

if __name__ == '__main__':
    main()
