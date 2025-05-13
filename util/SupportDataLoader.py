# *_*coding:utf-8 *_*
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from util.data_util import data_prepare


def rotate_point_cloud_z_with_normal(batch_xyz, feat, label):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
     
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    shape_pc = batch_xyz[:, :3]
    shape_normal = feat[:, :3]
    batch_xyz[:, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    feat[:, 0:3] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz, feat, label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class SemSegSupportDataset(Dataset):
    def __init__(self, root='/data/pcd_dental_texture', npoints=20000, split='train', shuffle=False, n_class=2, f_cols=3):
        self.npoints = npoints
        self.root = os.path.join(root, split)
        self.f_cols = f_cols
        print("load data!")
        npy_files = glob.glob(os.path.join(self.root, "*.npy"))
        # print("all_data: ", len(npy_files))
        # rot = ["X15", "X30", "X45", "X60", "X75", "X90", "Y15", "Y30", "Y45", "Y60", "Y75", "Y90"]
        # npy_files = [file for file in npy_files if file[-13:-10] not in rot]
        n_data = len(npy_files)
        print("ori_data: ", len(npy_files))
        if split != "test":
            label_weights = np.zeros(n_class)
            for npy_file in npy_files:
                data = np.load(npy_file).astype(np.float32) 
                data = data[data[:, 5] < 0]  # nz < 0
                # # print(npy_file, data.shape)
                # labels = data[:, -2].astype(np.int32)
                # labels[labels < 1] = 0
                # # labels[labels == 3] = 1
                # # labels[labels == 2] = 2
                # # labels[labels == 3] = 3
                labels = data[:, -1].astype(np.int32)  # 鑾峰彇鏍囩鍒?                # print(len(labels[labels > 0]))
                m_labels = data[:, -2].astype(np.int32)
                labels[(m_labels == 1) & (labels > 0)] = 1
                labels[(m_labels == 2) & (labels > 0)] = 2
                labels[(m_labels == 3) & (labels > 0)] = 3 
                tmp, _ = np.histogram(labels, range(n_class + 1))
                label_weights += tmp
            print("label_weights: ", label_weights)
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = np.amax(label_weights) / label_weights
            print("label_weights: ", self.label_weights)

        if shuffle:
            random.shuffle(npy_files)  # 闅忔満鎵撲贡 

        self.data_path = npy_files

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'toothModel': [0, 1, 2, 3]}

    def __getitem__(self, index):
        fn = self.data_path[index] 
        data = np.load(fn).astype(np.float32)
        # print(data.shape)
        data = data[data[:, 5] < 0]  # remove nz > 0
        # print("---", data.shape)
        np.random.shuffle(data)
        data = data[:self.npoints, :]
        # print(fn, data.shape)
        point_set = data[:, 0:self.f_cols]

        # labels = data[:, -2].astype(np.int32)
        # labels[labels < 1] = 0
        # labels[labels == 3] = 1
        labels = data[:, -1].astype(np.int32)  # 鑾峰彇鏍囩鍒?        # print(len(labels[labels > 0]))
        m_labels = data[:, -2].astype(np.int32)
        labels[(m_labels == 1) & (labels > 0)] = 1
        labels[(m_labels == 2) & (labels > 0)] = 2
        labels[(m_labels == 3) & (labels > 0)] = 3 
        coord = pc_normalize(point_set[:, 0:3]) 
        feat  = pc_normalize(point_set[:, 3:self.f_cols])  
        # print("after_pc_normalize:", point_set.shape, label.shape)
        if np.random.choice([0, 1]):
            coord, feat, labels = rotate_point_cloud_z_with_normal(coord, feat, labels)
        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = torch.FloatTensor(coord)
        feat = torch.FloatTensor(feat)
        label = torch.LongTensor(labels) 
        return coord, feat, label

    def __len__(self):
        return len(self.data_path)


def my_collate_fn_sem(batch_data):
    """
    descriptions: 瀵归綈鎵归噺鏁版嵁缁村害, [(data, label),(data, label)...]杞寲鎴?[data, data...],[label,label...])
    :param batch_data:  list锛孾(data, label),(data, label)...]
    :return: tuple, ([data, data...],[label,label...])
    """
    batch_data.sort(key=lambda x: len(x[0]), reverse=False)  # 鎸夌収鏁版嵁闀垮害鍗囧簭鎺掑簭
    data_list = []
    label_list = []
    min_len = len(batch_data[0][0])
    for batch in range(0, len(batch_data)):
        data = batch_data[batch][0]
        label = batch_data[batch][1]
        choice = np.random.choice(range(0, len(data)), min_len, replace=False)

        data = data[choice, :]
        label = label[choice]
        data_list.append(data)
        label_list.append(label)

    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    data_copy = (data_tensor, label_tensor)
    return data_copy


def my_collate_fn(batch_data):
    """
    descriptions: 瀵归綈鎵归噺鏁版嵁缁村害, [(data, label),(data, label)...]杞寲鎴?[data, data...],[label,label...])
    :param batch_data:  list锛孾(data, label),(data, label)...]
    :return: tuple, ([data, data...],[label,label...])
    """
    batch_data.sort(key=lambda x: len(x[0]), reverse=False)  # 鎸夌収鏁版嵁闀垮害鍗囧簭鎺掑簭
    data_list = []
    cls_list = []
    label_list = []
    min_len = len(batch_data[0][0])
    for batch in range(0, len(batch_data)):
        data = batch_data[batch][0]
        label = batch_data[batch][1] 
        cls = batch_data[batch][2]

        choice = np.random.choice(range(0, len(data)), min_len, replace=False)
        data = data[choice, :]
        label = label[choice]

        data_list.append(data)
        cls_list.append(cls)
        label_list.append(label)

    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    cls_tensor = torch.tensor(cls_list, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    data_copy = (data_tensor, label_tensor, cls_tensor)
    return data_copy


