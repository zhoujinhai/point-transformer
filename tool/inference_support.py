import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

import sys 
sys.path.append("/home/heygears/jinhai_zhou/learn/point-transformer/")

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize

random.seed(123)
np.random.seed(123)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/support/support_pt1_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--model_path', type=str, default="exp/support/pt1_repro/model/model_best_0.9.pth", help='model path', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser() 
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes).cuda()
    # logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    print("model_path: ", args.model_path)
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criterion, names)


def data_prepare():
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    elif args.data_name == 'support':
        data_list = Path(args.test_data_root).glob("*point.npy") 
    else:
        raise Exception('dataset not supported yet'.format(args.data_name)) 
    return data_list


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def data_load(data_name, f_cols):
    data = np.load(data_name).astype(np.float32) 
    ori_normal_data = data[data[:, 5] >= 0][:, :4]  
    ori_normal_data[:, -1] = 0  
    # print(data.shape)
    data = data[data[:, 5] < 0]  # remove nz > 0
    # print("---", data.shape)
    np.random.shuffle(data)
    ori_neg_data = data[:, :3]
    # print(fn, data.shape)
    point_set = data[:, 0:f_cols]

    label = data[:, -2].astype(np.int32)
    label[label < 1] = 0
    # label[label == 3] = 1
    coord = pc_normalize(point_set[:, 0:3]) 
    feat  = pc_normalize(point_set[:, 3:f_cols])  
    # print("after_pc_normalize:", point_set.shape, label.shape)

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label) 

    idx_data = []
    idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data, ori_normal_data, ori_neg_data


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    feat = feat / 255.
    return coord, feat


def show_pcl_data(data, label_cls=-1):
    import vedo
    points = data[:, 0:3] 

    colours = ["grey", "red", "blue", "yellow", "brown", "green", "black", "pink"]
    labels = data[:, label_cls]  # 鏈€鍚庝竴鍒椾负鏍囩鍒?    diff_label = np.unique(labels)
    print("res_label: ", diff_label)
    group_points = []
    group_labels = []
    for label in diff_label:
        point_group = points[labels == label]
        group_points.append(point_group)
        # print(point_group.shape)
        group_labels.append(label)

    show_pts = []
    for i, point in enumerate(group_points):
        pt = vedo.Points(point.reshape(-1, 3)).c((colours[int(group_labels[i]) % len(colours)]))  # 鏄剧ず鐐?        show_pts.append(pt)
    vedo.show(show_pts)


def test(model, criterion, names):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>') 
    args.batch_size_test = 1
    model.eval()
    
    data_list = data_prepare()
    for idx, item in enumerate(data_list): 
        if "XX8V3_VS_SET_VSc1_Subsetup1_Maxillar__X90_point" not in item.name:   # "shouban_kedaya_0_point" 
            continue 
        coord, feat, label, idx_data, ori_normal_data, ori_neg_data = data_load(item, args.fea_dim)
        print(coord.shape, feat.shape, label.shape, idx_data)
        
        idx_size = len(idx_data)
        idx_list, coord_list, feat_list, offset_list  = [], [], [], []
        for i in range(idx_size):
            idx_part = idx_data[i]
            coord_part, feat_part = coord[idx_part], feat[idx_part] 
            idx_list.append(idx_part), coord_list.append(coord_part), feat_list.append(feat_part), offset_list.append(idx_part.size)
        batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
        for i in range(batch_num):
            s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
            idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
            idx_part = np.concatenate(idx_part)
            coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
            feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
            offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
            with torch.no_grad():
                pred_part = model([coord_part, feat_part, offset_part])  # (n, k)
                print("pred_part: ", pred_part, pred_part.shape)
                probs = torch.exp(pred_part)
                print(probs.shape)
                max_values, max_indices = torch.max(probs, dim=1)
                print(max_values, max_values.shape, max_indices, max_indices.shape)
                pred_labels = max_indices.cpu().numpy()
                # nms_3d_point_cloud(ori_normal_neg_data[:, :3], labels, max_values[0].cpu().numpy())
                show_data = np.c_[ori_neg_data, pred_labels]
                print(show_data.shape, ori_normal_data.shape)
                show_data = np.vstack((show_data, ori_normal_data))
                print(show_data, show_data.shape)
                show_pcl_data(show_data)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
