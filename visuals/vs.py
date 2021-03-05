import cv2,h5py
import json
import random
import numpy as np
import os
from tqdm import tqdm

from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib as mpl

mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

import utilities.config as config
import utilities.utils as utils


class Visual:
    """ Visualize the results
    """

    def __init__(self, results_path, finetune_results_path, img_path, dest_dir='visuals/%s' % (config.version),
                 show_words=True, skip_results=False, preprocess_results=True):
        if skip_results:
            answers_json = utils.get_file(split="val", answer=True)
            self.results = {idx: [item['question_id'], item['multiple_choice_answer']] for idx, item in
                            enumerate(answers_json)}
        else:
            self.results = []
            if not preprocess_results:
                with open(os.path.join(results_path, 'cross_results.json'), 'r') as f:
                    self.results = json.load(f)
                with open(os.path.join(results_path, 'qid_to_iq.json'), 'r') as f:
                    self.q_iq = json.load(f)
            else:
                questions_json = utils.get_file(split="val", question=True)
                answers_json = utils.get_file(split="val", answer=True)
                self.q_iq = {ques['question_id']: [ques['image_id'], ques['question'], ans['multiple_choice_answer'],
                                                   ans['question_type'], ans['answer_type']] for ques, ans in
                             zip(questions_json, answers_json)}
                with open(os.path.join(results_path, 'qid_to_iq.json'), 'w') as f:
                    json.dump(self.q_iq, f)

                with open(os.path.join(results_path, 'eval_results.json'), 'r') as f:
                    false_results = json.load(f)
                with open(os.path.join(finetune_results_path, 'eval_results.json'), 'r') as f:
                    true_results = json.load(f)
                for f_idx, f_ques in enumerate(false_results):
                    for t_idx, t_ques in enumerate(true_results):
                        if f_ques['question_id'] == t_ques['question_id']:
                            self.results.append({'question_id': t_ques['question_id'], 'true_answer': t_ques['answer'],
                                                 'false_answer': f_ques['answer'], 'true_idx': t_idx, 'false_idx': f_idx})
                print(len(self.results))
                with open(os.path.join(results_path, 'cross_results.json'), 'w') as f:
                    json.dump(self.results, f)

            self.train_img_ids = utils.load_imageid(img_path + "train2014")
            self.val_img_ids = utils.load_imageid(img_path + "val2014")

            with h5py.File(os.path.join(finetune_results_path, 'att_weights.h5'), 'r') as f:
                self.true_weight = f['weights'][:]
                self.true_spatials = f['spatials'][:]
                self.hints = f['hints'][:]
            with h5py.File(os.path.join(results_path, 'att_weights.h5'), 'r') as f:
                self.false_weight = f['weights'][:]
                self.false_spatials = f['spatials'][:]

        self.train_image_fmt = img_path + "train2014/COCO_train2014_%s.jpg"
        self.val_image_fmt = img_path + "val2014/COCO_val2014_%s.jpg"
        self.skip_results = skip_results
        self.dest_dir = dest_dir
        if not os.path.isdir(dest_dir + ''):
            os.system('mkdir -p ' + dest_dir)
        self.show_words = show_words
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.length = self.__len__()
        self.color = {"blue": (255, 0, 0), "yellow":(0, 255, 255), "green":(0, 255, 0), "red":(0, 0, 255)}

    def _write_img(self, img, q_id, mode='normal'):
        cv2.imwrite('{}/{}_{}.png'.format(self.dest_dir, str(q_id).zfill(12), mode), img * 255.0)
        return

    def __len__(self):
        return len(self.results) - 1

    def _att_visual(self, img, att_map, att_spatial, hint_score=None, answer='', type="", top=2):
        """
        Visualize the attention map on the image and save the visualization.
        """
        def draw_max_weightbox(map, color, shift=0.0):
            list_map = list(map)
            weight = max(list_map)
            idx = list_map.index(weight)
            list_map.pop(idx)
            spatial = att_spatial[idx]
            spatial[:2] -= shift
            spatial[2:] += shift
            cv2.rectangle(img, (spatial[0], spatial[1]), (spatial[2], spatial[3]), self.color[color], 2)
            cv2.putText(img, str(round(weight,2)), (spatial[0], spatial[1]), self.font, 0.5, self.color["yellow"], 2)  # true green
            # box_width = spatial[2] - spatial[0]
            # box_height = spatial[3] - spatial[1]
            # print(weight)
            # xx, yy = int(spatial[0] // region_w), int(spatial[1] // region_h)
            # xw, yh = int(box_width // region_w), int(box_height // region_h)
            ## blend map box
            # demo = np.zeros([32, 32])
            # demo[yy: yy + yh, xx: xx + xw] = weight / 2.0
            return list_map, weight, idx

        img_h, img_w, img_c = img.shape
        # print(img_h, img_w)
        weights, shifts = [], []
        # att_map -= att_map.min()
        # if att_map.max() > 0:
        # att_map /= att_map.max()
        region_h = img_h // 32
        region_w = img_w // 32
        for i in range(top):

            att_map, weight, idx = draw_max_weightbox(att_map, "red")
            print("att", weight, idx)
            # print("hint", hint_score[idx], idx)
            hint_score, weight, idx = draw_max_weightbox(hint_score, "green", shift=20.0)
            print("hint", weight, idx)
            # weights.append(demo)
            # shifts.append(idx)

        # print(weight_min, weight_max)
        img = downsample_image(img)
        # for att_map in weights:
        #     img = get_blend_map(img, att_map)
        if self.show_words:
            img_h = img.shape[0]
            if type == "True":
                cv2.putText(img, answer, (20, img_h - 20), self.font, 0.7, (0, 255, 0), 2)  # true green
            elif type == "False":
                cv2.putText(img, answer, (20, img_h - 20), self.font, 0.7, (0, 0, 255), 2)  # false red
        return img

    def rcnn_cross_attention(self, sample_nums=0):
        mod = 1000 if config.version == 'v2' else 10
        if not sample_nums:
            sample_nums = self.length
        # samples = tqdm(random.sample(range(0, self.length), sample_nums), ncols=0)
        samples = tqdm(range(1000, sample_nums), ncols=0)
        for idx in samples:
            q_id = self.results[idx]['question_id']
            if q_id != 248744008:
                continue

            true_answer = self.results[idx]['true_answer']
            false_answer = self.results[idx]['false_answer']
            true_idx = self.results[idx]['true_idx']
            false_idx = self.results[idx]['false_idx']
            img_id, question, gta, qty, aty = self.q_iq[str(q_id)]
            if aty == "yes/no":
                continue
            if aty == "number":
                continue
            if img_id in self.train_img_ids:
                img_path = self.train_image_fmt % (str(img_id).zfill(12))
            elif img_id in self.val_img_ids:
                img_path = self.val_image_fmt % (str(img_id).zfill(12))
            else:
                print(img_id, 'error')
                continue
            print(img_path)
            assert ((self.true_spatials[true_idx] == self.false_spatials[false_idx]).all())
            # if qty != 'what color':
            #     continue
            img = cv2.imread(img_path)
            if self.show_words:
                cv2.putText(img, question, (20, 20), self.font, 0.7, self.color['blue'], 2)
                cv2.putText(img, gta, (20, 40), self.font, 0.7, self.color['green'], 2)
            else:
                print(question, gta, false_answer, true_answer)
            bb = self.true_spatials[true_idx]
            hint = self.hints[idx]
            amap = self.true_weight[idx][:, 0]
            self._write_img(self._att_visual(img.copy(), amap, bb, hint_score=hint*amap, answer=true_answer, type="True"), q_id, mode='True_Attention')
  
            # amap = self.false_weight[idx][:, 0]
            # self._write_img(self._att_visual(img.copy(), amap, bb, hint_score=hint*amap, answer=false_answer, type="False"), q_id,
            #                 mode='False_Attention')

            # amap = self.hints[idx]
            # self._write_img(self._att_visual(img.copy(), amap, bb, true_answer, type="Hint"), q_id, mode='Hint')

            self._write_img(downsample_image(img), q_id, mode='normal')
        return

    def rcnn_attention(self, sample_nums=0):
        if not sample_nums:
            sample_nums = self.length
        samples = tqdm(random.sample(range(0, self.length), sample_nums), ncols=0)
        for idx in samples:
            q_id, answer = self.results[idx]
            img_ids, questions, gta, qty = self.q_iq[q_id]
            if qty != 'what color':
                continue
            img_path = self.img_path_fmt % (str(img_ids).zfill(12))
            img = cv2.imread(img_path)
            if self.show_words:
                cv2.putText(img, questions, (20, 20), self.font, 0.7, (255, 0, 0), 2)
                cv2.putText(img, gta, (20, 40), self.font, 0.7, (0, 255, 0), 2)
            # bb = np.transpose(self.spatials[idx])
            # amap = self.weights[idx]
            # self._write_img(self._att_visual(img.copy(), amap, bb, answer, str(kd)), q_id, mode='att')
            self._write_img(downsample_image(img), q_id, mode='normal')

        # print('question:{}, predict_answer: {}, fact: {}'.format(questions,answer,str(kd)))
        return


def downsample_image(img):
    img_h, img_w, img_c = img.shape
    img = resize(img, (int(448 * img_h / img_w), 448), mode='constant', anti_aliasing=True)  # 22x22 regions
    return img


def get_blend_map(img, att_map, blur=True, overlap=True):
    # att_map -= att_map.min()
    att_map = resize(att_map, (img.shape[:2]), order=3, mode='constant', anti_aliasing=True)
    if blur:
        att_map = gaussian(att_map, 0.02 * max(img.shape[:2]))
    cmap = plt.get_cmap('jet')
    att_map_v = cmap(att_map)
    att_map_v = np.delete(att_map_v, 3, 2)
    if overlap:
        att_map = 1 * (1 - att_map ** 0.7).reshape(att_map.shape + (1,)) * img + (att_map ** 0.7).reshape(
            att_map.shape + (1,)) * att_map_v
    return att_map
