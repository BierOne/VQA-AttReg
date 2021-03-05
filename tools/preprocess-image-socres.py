import os,sys
sys.path.append(os.getcwd())
import h5py, pickle
from tqdm import tqdm
from utilities import config, utils

features_shape = (
    82783 + 40504,  # number of images in trainval or in test
    config.rcnn_output_size,
    config.output_features,
)
boxes_shape = (
    features_shape[0],
    config.rcnn_output_size,
    4,
)

cls_shape = (
    82783 + 40504,  # number of images in trainval or in test
    36,
    2,
)

if __name__ == '__main__':
    share_path = '/home/share/liuyibing'
    train_imgid2idx = pickle.load(open(os.path.join(share_path, 'train36_imgid2img.pkl'), 'rb'))
    train_idx2imgid = {idx: img_id for img_id, idx in train_imgid2idx.items()}
    val_imgid2idx = pickle.load(open(os.path.join(share_path, 'val36_imgid2img.pkl'), 'rb'))
    val_idx2imgid = {idx: img_id for img_id, idx in val_imgid2idx.items()}
    with h5py.File(os.path.join(share_path, 'train36.hdf5'), 'r') as hf:
        train_features = hf['image_features'][:]
        train_spatials = hf['spatial_features'][:]
        train_cls = hf['cls_score'][:]
        train_att = hf['attr_score'][:]
    with h5py.File(os.path.join(share_path, 'val36.hdf5'), 'r') as hf:
        val_features = hf['image_features'][:]
        val_spatials = hf['spatial_features'][:]
        val_cls = hf['cls_score'][:]
        val_att = hf['attr_score'][:]
    with h5py.File(config.rcnn_trainval_path, mode='w', libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float32')
        boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
        coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')
        widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
        heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')
        cls_scores = fd.create_dataset('cls_score', shape=cls_shape, dtype='float32')
        attr_scores = fd.create_dataset('attr_score', shape=cls_shape, dtype='float32')

        for new_idx in tqdm(range(82783)):
            img_id = train_idx2imgid[new_idx]
            coco_ids[new_idx] = int(img_id)
            features[new_idx, ...] = train_features[new_idx, ...]
            boxes[new_idx, ...] = train_spatials[new_idx, :, :4]
            widths[new_idx] = train_spatials[new_idx, 0, 6]
            heights[new_idx] = train_spatials[new_idx, 0, 5]
            cls_scores[new_idx, ...] = train_cls[new_idx][:]
            attr_scores[new_idx, ...] = train_att[new_idx][:]

        for new_idx in tqdm(range(82783, features_shape[0])):
            old_idx = new_idx - 82783
            img_id = val_idx2imgid[old_idx]
            coco_ids[new_idx] = int(img_id)
            features[new_idx, ...] = val_features[old_idx, ...]
            boxes[new_idx, ...] = val_spatials[old_idx, :, :4]
            widths[new_idx] = val_spatials[old_idx, 0, 6]
            heights[new_idx] = val_spatials[old_idx, 0, 5]
            cls_scores[new_idx, ...] = val_cls[old_idx][:]
            attr_scores[new_idx, ...] = val_att[old_idx][:]

    ## add datasets: cls_socre and attr_score
    # with h5py.File(config.rcnn_trainval_path, 'a') as hf:
    #     total_img_id2idx = {img_id: idx for idx, img_id in enumerate(hf.get('ids'))}
    #     if 'cls_score' in list(hf.keys()):
    #         del hf['cls_score']
    #         del hf['attr_score']
    #     new_cls_scores = hf.create_dataset('cls_score', shape=shape, dtype='float32')
    #     new_attr_scores = hf.create_dataset('attr_score', shape=shape, dtype='float32')
    #
    #     for i, img_id in enumerate(tqdm(total_img_id2idx.keys())):
    #         new_idx = total_img_id2idx[img_id]
    #         if img_id in train_imgid2img.keys():
    #             new_cls_scores[new_idx, ...] = train_cls[train_imgid2img[img_id]][:]
    #             new_attr_scores[new_idx, ...] = train_att[train_imgid2img[img_id]][:]
    #         elif img_id in val_imgid2img.keys():
    #             new_cls_scores[new_idx, ...] = val_cls[val_imgid2img[img_id]][:]
    #             new_attr_scores[new_idx, ...] = val_att[val_imgid2img[img_id]][:]
    #         else:
    #             raise Exception('img_id not in train or val')
