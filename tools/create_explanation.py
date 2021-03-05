import os, sys

import argparse
import numpy as np
import json, pickle

sys.path.append(os.getcwd())
from utilities.dataset import Dictionary
from utilities import config, utils


def create_vqa_exp_dictionary():
    dictionary = Dictionary()
    qid2exp = {}
    if config.type == 'cp':
        ques_file_types = ['train', 'test']
    else:
        ques_file_types = ['train', 'val']
    for type in ques_file_types:
        questions = utils.get_file(type, question=True)
        questions = sorted(questions, key=lambda x: x['question_id'])
        answers = utils.get_file(type, answer=True)
        answers = sorted(answers, key=lambda x: x['question_id'])
        for q, a in zip(questions, answers):
            ques_id = a.pop('question_id')
            utils.assert_eq(q['question_id'], ques_id)
            dictionary.tokenize(q['question'], True)
            dictionary.tokenize(a['multiple_choice_answer'], True)
            qid2exp[int(ques_id)] = {'question': q['question'], 'answer': a['multiple_choice_answer']}
        print(ques_id, a['image_id'])
    return dictionary, qid2exp


def create_vqx_exp_dictionary():
    dictionary = Dictionary()
    path = os.path.join(config.main_path, 'VQA-X/textual/')
    qid2exp = {}
    files = ['test_exp_anno.json', 'train_exp_anno.json', 'val_exp_anno.json']
    for file in files:
        file_path = os.path.join(path, file)
        exps = json.load(open(file_path))
        for qid in exps.keys():
            # choose the first textual explanation
            dictionary.tokenize(exps[qid][0], True)
            qid2exp[int(qid)] = exps[qid][0]
    print('%s finished'%file)
    return dictionary, qid2exp


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word) + 1, emb_dim), dtype=np.float32)  # padding

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)

    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='vqx', help='explanation from vqx-exp or vqa-qa')
    parser.add_argument('--emb_dim', type=int, default=300, help='glove embedding dim')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.exp == 'vqx':
        d, qid2exp = create_vqx_exp_dictionary()
        d.dump_to_file(os.path.join(config.main_path, 'vqx_dictionary.json'))
        d = Dictionary.load_from_file(os.path.join(config.main_path, 'vqx_dictionary.json'))
        pickle.dump(qid2exp, open(os.path.join(config.main_path, 'vqx_qid2exp.pkl'), 'wb'))
    else:
        d, qid2exp = create_vqa_exp_dictionary()
        d.dump_to_file(os.path.join(config.cache_root, 'vqa_caption_dictionary.json'))
        d = Dictionary.load_from_file(os.path.join(config.cache_root, 'vqa_caption_dictionary.json'))
        pickle.dump(qid2exp, open(os.path.join(config.cache_root, 'vqa_qid2exp.pkl'), 'wb'))

    glove_file = os.path.join(config.glove_path, 'glove.6B.{}d.txt'.format(args.emb_dim))
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)

    if args.exp == 'vqx':
        glove_name = 'glove6b_init_vqx_{}d.npy'.format(args.emb_dim)
        np.save(os.path.join(config.main_path, glove_name), weights)
    else:
        glove_name = 'glove6b_init_vqa_caption_{}d.npy'.format(args.emb_dim)
        np.save(os.path.join(config.cache_root, glove_name), weights)


if __name__ == '__main__':
    main()
