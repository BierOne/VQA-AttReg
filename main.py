import os, argparse
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.optim.lr_scheduler import ExponentialLR

from model import base_model
from utilities import config, utils, dataset as data

# from train.train_baseline_with_mask import run

from train.train_baseline_to_get_importance import run as run_importance


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.01)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='temp')
    parser.add_argument('--opt_step', type=str, default='two', help='one or two')
    parser.add_argument('--resume', action='store_true', help='resumed flag')
    parser.add_argument('--test', action='store_true', dest='test_only')
    parser.add_argument('--save_every', action='store_true', dest='save_every')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=28, help='random seed')
    parser.add_argument('--gpu', default='0', help='the chosen gpu id')
    parser.add_argument('--lamda', type=float, default=0, help='lamda')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--mask_ratio', type=float, default=0, help='mask_ratio')
    parser.add_argument('--mask_type', type=str, default='att', help='att or grad')
    parser.add_argument('--optimizer', type=str, default='adamax',
                        help='what update to use? rmsprop|adamax|adadelta|adam')
    parser.add_argument('--get_importance', action='store_true', help='run baseline_to_get_importance')
    parser.add_argument('--pattern', type=str, default='baseline', help='baseline|finetune')
    parser.add_argument('--vs', action='store_true', help='visualize the attention map')
    args = parser.parse_args()
    return args


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("seed: ", seed)


def saved_for_eval(eval_loader, result, output_path, epoch=None, vs=False):
    """
        save a results file in the format accepted by the submission server.
        input result : [ans_idxs, acc, q_ids, att_weights, spatials, mask_ans_idxs]
    """
    label2ans = eval_loader.dataset.label2ans
    results, mask_results = [], []
    for a, q_id in zip(result[0], result[2]):
        results.append({'question_id': int(q_id), 'answer': label2ans[a]})
        # mask_results.append({'question_id': int(q_id), 'answer': label2ans[ma]})
    if config.mode != 'trainval':
        name = 'eval_results.json'
    else:
        name = 'eval_results_epoch{}.json'.format(epoch)
        # results_path = os.path.join(output_path, name)
        # weights_path = os.path.join(output_path, 'att_weights.h5')
        # mask_results_path = os.path.join(output_path, 'mask_{}_eval_results.json'.format(config.masks))
        # # mask_results_path = os.path.join(output, 'att_random_eval_results.json')
        # with open(mask_results_path, 'w') as fd:
        #     json.dump(mask_results, fd)
    if vs:
        path = os.path.join(output_path, "vs/")
        utils.create_dir(path)
        with open(os.path.join(path, name), 'w') as fd:
            json.dump(results, fd)
        with h5py.File(os.path.join(path, 'att_weights.h5'), 'w') as f:
            f.create_dataset('weights', data=result[3])
            f.create_dataset('spatials', data=result[4])
            f.create_dataset('hints', data=result[6])
    else:
        with open(os.path.join(output_path, name), 'w') as fd:
            json.dump(results, fd)

if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed)
    print("epochs", args.epochs)
    print("use_debias", config.use_debias)
    print("use_rubi", config.use_rubi)
    print("use_hint", config.use_hint)
    print("optimize_type", config.optimize_type)
    print("mask_type", args.mask_type)
    print("use_rho", config.use_rho)
    print("lamda_nf", args.lamda)
    print("lr", args.lr)
    print("masks", config.masks)
    print("num_sub", config.num_sub)
    print("att_norm", config.att_norm)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.test_only:
        args.resume = True

    # from train.train_baseline_with_mask import run
    if args.pattern == 'baseline':
        from train.train_baseline import run
    else:
        from train.train_with_att_debias import run

    output_path = 'saved_models/{}/{}'.format(config.type + config.version, args.output)
    utils.create_dir(output_path)
    torch.backends.cudnn.benchmark = True

    ######################################### DATASET PREPARATION #######################################
    if config.mode == 'train':
        train_loader = data.get_loader('train')
        val_loader = data.get_loader('val')
    elif args.test_only:
        # train_loader = data.get_loader('trainval')
        val_loader = data.get_loader('test')
    else:
        train_loader = data.get_loader('trainval')
        val_loader = data.get_loader('test')

    num_ans_candidates = val_loader.dataset.num_ans_candidates
    if config.use_debias and config.mode != 'trainval':
        utils.cp_bias_from_trainset_to_valset(train_loader, val_loader)
    ######################################### MODEL PREPARATION #######################################
    embeddings = np.load(os.path.join(config.cache_root, 'glove6b_init_300d.npy'))
    constructor = 'build_baseline_with_%sstep' % args.opt_step
    model = getattr(base_model, constructor)(embeddings, num_ans_candidates).cuda()
    # model.apply(weights_init_kn)
    model = nn.DataParallel(model).cuda()

    if args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0,
                                    centered=False)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizers = {'all_optimizer': optimizer }

    r = np.zeros(3)
    start_epoch = 0
    acc_val_best = 0
    tracker = utils.Tracker()
    model_path = os.path.join(output_path, 'model.pth')
    if args.resume:
        logs = torch.load(model_path)
        start_epoch = logs['epoch']
        model.load_state_dict(logs['model_state'])
        if args.pattern != 'finetune':
            acc_val_best = logs['acc_val_best']
            for k, v in logs['optim_state'].items():
                optimizers[k].load_state_dict(v)

    ######################################### Attention GradCAM Measure #######################################
    # if args.get_importance == True:
    #     with h5py.File('./cpv2_img_importance.h5', 'a') as fd:
    #         importance = fd.create_dataset('importance', shape=(len(val_loader.dataset), 36), dtype='float32')
    #         diff_importance = fd.create_dataset('diff_importance', shape=(len(val_loader.dataset), 36), dtype='float32')
    #         diff_importance_abs = fd.create_dataset('diff_importance_abs', shape=(len(val_loader.dataset), 36), dtype='float32')
    #         for i in range(36):
    #             r = run_importance(model, val_loader, optimizers, tracker, train=False,
    #                     prefix='val', epoch=i, has_answers=(config.mode == 'train'), mask_pos= i)
    #             importance[:, i] = r[1] # delta_acc
    #             diff_importance[:, i] = r[7] # delta_acc
    #             diff_importance_abs[:, i] = r[8]  # delta_acc
    #         fd.create_dataset('attw', data=r[3])
    #         fd.create_dataset('gradw', data=r[6])
    ######################################### MODEL RUN #######################################
    if not args.test_only:
        for epoch in range(start_epoch, args.epochs):
                run(model, train_loader, optimizers, tracker, train=True, prefix='train', epoch=epoch, args=args)
                if not (config.mode == 'trainval' and epoch in range(args.epochs - 5)):
                    r = run(model, val_loader, optimizers, tracker, train=False,
                                   prefix='val', epoch=epoch, has_answers=(config.mode == 'train'), args=args)
                results = {
                    'epoch': epoch,
                    'acc_val_best': acc_val_best,
                    'model_state': model.state_dict(),
                    'optim_state': {k: v.state_dict() for k, v in optimizers.items()},
                }
                if args.save_every == True:
                    torch.save(results, model_path)
                    continue
                if config.mode == 'train' and r[1].mean() > acc_val_best:
                    # if not args.resume:
                    #     torch.save(results, model_path)
                    torch.save(results, model_path)
                    acc_val_best = r[1].mean()
                    saved_for_eval(val_loader, r, output_path, epoch, args.vs)
                if config.mode == 'trainval' and epoch in range(args.epochs - 5, args.epochs):
                    saved_for_eval(val_loader, r, output_path, epoch)
                    torch.save(results, model_path)
    else:
        r = run(model, val_loader, optimizers, tracker, train=False,
                prefix='test', epoch=start_epoch, has_answers=(config.mode == 'train'), args=args)
        saved_for_eval(val_loader, r, output_path, start_epoch, args.vs)