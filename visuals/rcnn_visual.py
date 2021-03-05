import sys
# from visualize.visual import *
import h5py
import json
import random

import numpy as np
import os
from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import cv2

font=cv2.FONT_HERSHEY_SIMPLEX
data_type = 'v2_'
false_results_dir = "vqa-counting/vqa-counting/vqa-v2/"
true_results_dir = "vqa-counting/prior-counting/"
# with open('%s/results.json'%false_results_dir,'r') as f:
	# false_results = json.load(f)
# with open('%s/results.json'%true_results_dir,'r') as f:
	# true_results  = json.load(f)
with open('visualize/counter_results_epoch20.json','r') as f:
	cross_results  = json.load(f)

Question_dir = "/home/share/guoyangyang/vqa/data"
with open('%s/%sOpenEnded_mscoco_val2014_questions.json'%(Question_dir, data_type),'r') as f:
	data = json.load(f)
	qi =  {ques['question_id']: ques['image_id'] for ques in data['questions']}
	qq =  {ques['question_id']: ques['question'] for ques in data['questions']}

false_weights = h5py.File('%s/weights.h5'%false_results_dir, 'r')
true_weights = h5py.File('%s/weights.h5'%true_results_dir, 'r')
# print( len(false_results))
def get_cross_results(start,end):
	cross_results = {}
	for i in range(start,end):
		false_res = false_results[str(i)]
		true_res = true_results[str(i)]
		cross_results[i] = []
		for f_idx, f_ques in enumerate(false_res):
			for t_idx, t_ques in enumerate(true_res):
				if f_ques['question_id'] == t_ques['question_id']:
					cross_results[i].append({'question_id': t_ques['question_id'], 'true_answer':t_ques['answer'], 'false_answer': f_ques['answer'],'true_idx':t_idx, 'false_idx':f_idx })
		print("The cross_results: epoch:{}, length: {}".format(i,len(cross_results[i])))
	return cross_results


def get_blend_map(img, att_map, weight_min=0, weight_max=0, blur=True, overlap=True):
	# att_map -= att_map.min()
	# if att_map.max() > 0:
		# att_map /= att_map.max()
	att_map = resize(att_map, (img.shape[:2]), order = 3, mode='constant', anti_aliasing=True)
	if blur:
		att_map = gaussian(att_map, 0.02*max(img.shape[:2]))
		# att_map -= att_map.min()
		# att_map /= att_map.max()
	cmap = plt.get_cmap('jet')
	att_map_v = cmap(att_map)
	att_map_v = np.delete(att_map_v, 3, 2)
	if overlap:
		att_map = 1*(1-att_map**0.7).reshape(att_map.shape + (1,))*img + (att_map**0.7).reshape(att_map.shape+(1,)) * att_map_v
	return att_map


def downsample_image(img):
	img_h, img_w, img_c = img.shape
	img = resize(img, (int(448 * img_h / img_w), 448), mode='constant', anti_aliasing=True) #22x22 regions
	return img


def save_attention_visualization(top, img, att_map, att_spatial, answer, file_name="COCO_xxx", dest_dir = "visuals/", type=False):
	"""
	Visualize the attention map on the image and save the visualization.
	"""
	path0 = os.path.join(dest_dir, file_name + '.png')
	img_h, img_w, img_c = img.shape
	# print(img_h, img_w)
	
	weights = []
	# att_map -= att_map.min()
	# if att_map.max() > 0:
		# att_map /= att_map.max()
	att_map = list(att_map)
	
	region_h = img_h//32
	region_w = img_w//32
	for i in range(top):
		weight = max(att_map)
		idx = att_map.index(weight)
		att_map.pop(idx)
		spatial = att_spatial[idx]
		box_width = spatial[2] - spatial[0]
		box_height = spatial[3] - spatial[1]
		# print(weight)
		# scaled_w = box_width / img_w
		# scaled_h = box_height / img_h
		# scaled_x = spatial[0] / img_w
		# scaled_y = spatial[1] / img_h
		xx = int(spatial[0]//region_w)
		yy = int(spatial[1]//region_h)
		xw = int(box_width//region_w)
		yh = int(box_height//region_h)
		# print(xx, yy, xw, yh)
		demo = np.zeros([32, 32])
		demo[yy: yy+yh, xx: xx+xw] = weight
		cv2.rectangle(img, (spatial[0], spatial[1]), (spatial[2], spatial[3]), (0, 0, 255), 1)
		weights.append(demo)
	# print(weight_min, weight_max)
		# print(xx, yy)
	img = downsample_image(img)
	for att_map in weights:
		img = get_blend_map(img, att_map)
	
	img_h = img.shape[0]

	################################write message ###################################
	# if type:
		# cv2.putText(img, answer, (20,img_h-20), font, 0.7, (0,255,0), 2) #true green
	# else:
		# cv2.putText(img, answer, (20,img_h-20), font, 0.7, (0,0,255), 2) #false red
	#################################################################################

	cv2.imwrite(path0, img *255.0)
	return


def getImgIds(quesIds=[]):
	# quesIds	= quesIds	if type(quesIds)   == list else [quesIds]
	# img_ids = [qi[quesId] for quesId in quesIds]
	img_ids = qi[quesIds]
	return img_ids, qq[quesIds]


def rcnn_attention(sample_nums, results, false_weights, false_spatials, true_weights, true_spatials, dir):
	# print(len(results), len(false_weights), len(false_weights[0]), len(true_spatials), len(true_spatials[0]))
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
		
		# if true_answer=='yes' or true_answer=='no':
			# continue
		# if q_id != 545101001 and q_id != 519853003 and q_id != 477216008: 
			# continue
		print('question:{}, false_answer: {}, true_answer: {} img_id: {}'.format(questions, false_answer, true_answer, img_ids))
		source_img_path = '/home/share/guoyangyang/vqa/mscoco/val2014/COCO_val2014_%s.jpg'
		img_path = source_img_path%(str(img_ids).zfill(12))

		img = cv2.imread(img_path)
		# cv2.putText(img, questions, (20,20), font, 0.7, (255,0,0), 2)

		assert((false_spatials[f_idx] == true_spatials[t_idx]).all())
		demo_bb = np.transpose(false_spatials[f_idx])
		
		file_name='%s_False_Attention'%(str(q_id).zfill(12))
		demo_map = false_weights[f_idx]
		save_attention_visualization(3, img.copy(), demo_map, demo_bb, false_answer, file_name, dest_dir )

		file_name='%s_True_Attention'%(str(q_id).zfill(12))
		demo_map = true_weights[t_idx]
		save_attention_visualization(3, img.copy(), demo_map, demo_bb, true_answer, file_name, dest_dir, type=True )

		img = downsample_image(cv2.imread(img_path)) # cv2.imread does auto-rotate
		cv2.imwrite(dest_dir+ '%s_Normal.png'%(str(q_id).zfill(12)), img*255.0)
	return

def main():
	start = 19
	end = 20
	model = 'counter'
	# cross_results = get_cross_results(start,end)
	# with open('./visualize/counter_results_epoch20.json','w') as f:
		# json.dump(cross_results,f)
	# print('dump ok !')
	for i in range(start, end):
		rcnn_attention(len(cross_results[str(i)])-1, cross_results[str(i)], false_weights['epoch_%d_weights'%(i)], false_weights['epoch_%d_spatials'%(i)], true_weights['epoch_%d_weights'%i], true_weights['epoch_%d_spatials'%i], dir='%s/results_%s'%(model, i))


if __name__ == '__main__':
	main()