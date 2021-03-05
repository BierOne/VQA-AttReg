import os, sys

import argparse
import numpy as np
import json, pickle

sys.path.append(os.getcwd())
from utilities.dataset import Dictionary
from utilities import config, utils


def create_vqa_dictionary():
    dictionary = Dictionary()
    if config.type == 'cp':
        ques_file_types = ['train', 'test']
    else:
        ques_file_types = ['train', 'val', 'test']
    for type in ques_file_types:
        questions = utils.get_file(type, question=True)
        for q in questions:
            dictionary.tokenize(q['question'], True)
    return dictionary


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
    parser.add_argument('--emb_dim', type=int, default=300, help='glove embedding dim')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    d = create_vqa_dictionary()
    d.dump_to_file(os.path.join(config.cache_root, 'dictionary.json'))
    d = Dictionary.load_from_file(os.path.join(config.cache_root, 'dictionary.json'))

    glove_file = os.path.join(config.glove_path, 'glove.6B.{}d.txt'.format(args.emb_dim))
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)

    glove_name = 'glove6b_init_{}d.npy'.format(args.emb_dim)
    np.save(os.path.join(config.cache_root, glove_name), weights)


if __name__ == '__main__':
    main()
