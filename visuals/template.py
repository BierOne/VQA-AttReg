import cv2
import numpy as np
import sys, os
from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib.pyplot as plt


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
	return img


def save_attention_visualization(img, att_map, file_name="COCO_xxx", dest_dir = "visuals/"):
	"""
	Visualize the attention map on the image and save the visualization.
	"""
	path0 = os.path.join(dest_dir, file_name + '.png')
	# att_h, att_w = att_map.shape
	# att_map = att_map.reshape((att_h, att_w))
	heat_map = get_blend_map(img, att_map)
	cv2.imwrite(path0, heat_map *255.0)
	return


def main():
	source_img_path = '/home/share/liuyibing/vqa/mscoco/val2014/COCO_val2014_%s.jpg'
	img_path = source_img_path%(str(img_ids).zfill(12))
	img = downsample_image(cv2.imread(img_path)) # cv2.imread does auto-rotate
	save_attention_visualization(img, demo_map.reshape(14, 14), 'xxx')

if __name__ == '__main__':
	main()