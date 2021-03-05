import sys
# from visualize.visual import *
import h5py
import json
import random
import cv2
import numpy as np
import os
from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from scipy import misc


font=cv2.FONT_HERSHEY_SIMPLEX
false_results_dir = "vqa-mfb/mfb_coatt_glove/false_temp/"
true_results_dir = "prior-mfb/mfb_coatt_glove/"
# with open('%s/results.json'%false_results_dir,'r') as f:
	# false_results = json.load(f)
# with open('%s/results.json'%true_results_dir,'r') as f:
	# true_results  = json.load(f)
with open('visualize/mfb_epoch10_cross_results.json','r') as f:
	cross_results  = json.load(f)
Question_dir = "/home/share/guoyangyang/vqa/data"
with open('%s/OpenEnded_mscoco_val2014_questions.json'%Question_dir,'r') as f:
	data = json.load(f)
	qi =  {ques['question_id']: ques['image_id'] for ques in data['questions']}
	qq =  {ques['question_id']: ques['question'] for ques in data['questions']}

false_weights = h5py.File('%s/weights.h5'%false_results_dir, 'r')
true_weights = h5py.File('%s/weights.h5'%true_results_dir, 'r')

def get_cross_results(start,end):
	cross_results = {}
	for i in range(start,end):
		# false_res = false_results[str(i)]
		# true_res = true_results[str(i)]
		false_res = false_results
		true_res = true_results
		cross_results[i] = []
		for f_idx, f_ques in enumerate(false_res):
			for t_idx, t_ques in enumerate(true_res):
				if f_ques['question_id'] == t_ques['question_id']:
					cross_results[i].append({'question_id': t_ques['question_id'], 'true_answer':t_ques['answer'], 'false_answer': f_ques['answer'],'true_idx':t_idx, 'false_idx':f_idx })
		print("The cross_results: epoch:{}, length: {}".format(i,len(cross_results[i])))
	return cross_results


def get_blend_map(img, att_map, blur=True, overlap=True):
	att_map -= att_map.min()
	if att_map.max() > 0:
		att_map /= att_map.max()
	att_map = resize(att_map, (img.shape[:2]), order = 3, mode='constant', anti_aliasing=True)
	if blur:
		att_map = gaussian(att_map, 0.02*max(img.shape[:2]))
		att_map -= att_map.min()
		att_map /= att_map.max()
	cmap = plt.get_cmap('jet')
	att_map_v = cmap(att_map)
	att_map_v = np.delete(att_map_v, 3, 2)
	if overlap:
		att_map = 1*(1-att_map**0.7).reshape(att_map.shape + (1,))*img + (att_map**0.7).reshape(att_map.shape+(1,)) * att_map_v
	return att_map


def downsample_image(img):
	img_h, img_w, img_c = img.shape
	img = resize(img, (int(448 * img_h / img_w), 448), mode='constant', anti_aliasing=True)
	# 22x22 regions
	# img = misc.imresize(img, (300, 300), interp='bicubic')
	return img


def save_attention_visualization(img, att_map, answer, file_name="COCO_xxx", dest_dir = "visuals/", type=False):
	"""
	Visualize the attention map on the image and save the visualization.
	"""
	path0 = os.path.join(dest_dir, file_name + '.png')
	img_h, img_w, img_c = img.shape
	att_h, att_w = att_map.shape
	att_map = att_map.reshape((att_h, att_w))
	heat_map = get_blend_map(img, att_map)
	# if type:
		# cv2.putText(heat_map, answer, (20,200), font, 0.7, (0,255,0), 2)
	# else:
		# cv2.putText(heat_map, answer, (20,200), font, 0.7, (0,0,255), 2)
	cv2.imwrite(path0, heat_map *255.0)
	return


def getImgIds(quesIds=[]):
	img_ids = qi[quesIds]
	return img_ids, qq[quesIds]


def visual_attention(sample_nums, results, false_weights, true_weights, dir, map_shape=[14,14]):
	print(len(results), len(true_weights), len(true_weights[0]), np.sum(true_weights[0]))
	samples = random.sample(range(0,len(results)-1),sample_nums)
	dest_dir='visualize/%s/'%(dir)
	if not os.path.isdir(dest_dir+''):
		os.system('mkdir -p ' + dest_dir)
	for idx in samples:
		q_id = results[idx]['question_id']
		img_ids, questions = getImgIds(quesIds=q_id)
		true_answer = results[idx]['true_answer']
		t_idx = results[idx]['true_idx']
		false_answer = results[idx]['false_answer']
		f_idx = results[idx]['false_idx']
		
		if false_answer=='yes' or  false_answer=='no':
			continue
		if q_id != 618361 and q_id != 29850:
			continue
		print('question:{}, false_answer: {}, true_answer: {} img_id: {} \n'.format(questions, false_answer, true_answer, img_ids))
		source_img_path = '/home/share/guoyangyang/vqa/mscoco/val2014/COCO_val2014_%s.jpg'
		img_path = source_img_path%(str(img_ids).zfill(12))

		img = downsample_image(cv2.imread(img_path)) # cv2.imread does auto-rotate
		# cv2.putText(img, questions, (20,20), font, 0.7, (255,0,0), 2)

		file_name='%s_False_Attention_val2014'%(str(q_id).zfill(12))
		demo_map = false_weights[f_idx]
		save_attention_visualization(img, demo_map.reshape(map_shape[0],map_shape[1]), false_answer, file_name,dest_dir )

		file_name='%s_True_Attention_val2014'%(str(q_id).zfill(12))
		demo_map = true_weights[t_idx]
		save_attention_visualization(img, demo_map.reshape(map_shape[0],map_shape[1]), true_answer, file_name, dest_dir,type=True)

		cv2.imwrite(dest_dir+ '%s_Normal.png'%(str(q_id).zfill(12)), img*255.0)

	print(dir,'ok!')
	return

def main():
	start = 0
	end = 1
	model = 'mfb'
	# cross_results = get_cross_results(start,end)
	# with open('./visualize/mfb_epoch10_cross_results.json','w') as f:
		# json.dump(cross_results,f)
	for i in range(start, end):
		visual_attention(len(cross_results[str(i)])-1, cross_results[str(i)], false_weights['epoch_%s'%i], true_weights['epoch_%s'%i], dir='%s/results_%s'%(model, i))

if __name__ == '__main__':
	main()
