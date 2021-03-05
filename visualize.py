import cv2
import utilities.config as config
import utilities.utils as utils
import visuals.vs as vs

train = False
val = True
def main():
	img_path = "/home/share/liuyibing/vqa/mscoco/"
	# results_path = './saved_models/{}/28_seed_rerun_baseline_debias/vs'.format(config.type + config.version)
	# finetune_results_path = './saved_models/{}/28_seed_rerun_baseline_debias_finetune/vs'.format(config.type + config.version)
	results_path = './saved_models/{}/28_seed_rerun_baseline/vs'.format(config.type + config.version)
	finetune_results_path = './saved_models/{}/28_seed_rerun_baseline_finetune/vs'.format(config.type + config.version)
	Visual = vs.Visual(
		results_path = results_path,
		finetune_results_path = finetune_results_path,
		img_path = img_path,
		dest_dir='./visuals/%s'%(config.version),
		show_words=False,
		skip_results=False,
		preprocess_results=False
	)
	
	Visual.rcnn_cross_attention(10000)
	return

if __name__ == '__main__':
	main()