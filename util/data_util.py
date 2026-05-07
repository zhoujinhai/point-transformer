import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)

def collate_fn_cls(batch):
    """
    分类任务的collate函数
    batch: 列表，每个元素是 (coord, feat, label) 元组
    返回: coords, feats, labels, offsets
    """
    # 解压batch
    coords, feats, labels = list(zip(*batch)) 
    
    # 计算offset（每个样本的点数偏移量）
    offset, count = [], 0
    for item in coords:
        count += item.shape[0]
        offset.append(count)
    
    # 拼接坐标和特征
    coords_tensor = torch.cat(coords)
    feats_tensor = torch.cat(feats)
    
    labels_list = []
    for lbl in labels:
        if torch.is_tensor(lbl):
            # 检查是否为空张量
            if lbl.numel() == 0:
                # 空张量，假设类别为0
                labels_list.append(0)
            else:
                # 非空张量，转换为标量
                labels_list.append(lbl.item())
        else:
            # 非张量，直接使用
            labels_list.append(lbl)
    
    # 创建标签张量
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    
    # offset转换为张量
    offset_tensor = torch.IntTensor(offset)
    # print(coords_tensor, labels_tensor, offset_tensor)
    return coords_tensor, feats_tensor, labels_tensor, offset_tensor
    
def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label
