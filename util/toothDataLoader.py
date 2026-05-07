# *_*coding:utf-8 *_*
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
# from util.data_util import data_prepare


class RandomRotate(object):
    def __init__(self, angle=[1, 1, 1]):
        self.angle = angle

    def __call__(self, coord, feat, label):
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        coord = np.dot(coord, np.transpose(R))
        feat[:, :3] = np.dot(feat[:, :3], np.transpose(R))
        return coord, feat, label


class RandomScale(object):
    def __init__(self, scale=[0.9, 1.1], anisotropic=False):
        self.scale = scale
        self.anisotropic = anisotropic

    def __call__(self, coord, feat, label):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        coord *= scale
        return coord, feat, label


class RandomShift(object):
    def __init__(self, shift=[0.2, 0.2, 0]):
        self.shift = shift

    def __call__(self, coord, feat, label):
        shift_x = np.random.uniform(-self.shift[0], self.shift[0])
        shift_y = np.random.uniform(-self.shift[1], self.shift[1])
        shift_z = np.random.uniform(-self.shift[2], self.shift[2])
        coord += [shift_x, shift_y, shift_z]
        return coord, feat, label


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


def safe_augmentation(original_data, augmentation_func, *args, **kwargs):
     
    coord_orig, feat_orig, labels_orig = original_data
    
    try: 
        coord_aug, feat_aug, labels_aug = augmentation_func(coord_orig.copy(), 
                                                           feat_orig.copy(), 
                                                           labels_orig.copy(), 
                                                           *args, **kwargs)
         
        if (np.isnan(coord_aug).any() or np.isinf(coord_aug).any() or
            np.isnan(feat_aug).any() or np.isinf(feat_aug).any()): 
            return coord_orig, feat_orig, labels_orig
        
        return coord_aug, feat_aug, labels_aug
    
    except Exception as e: 
        print(f" data aug: {e}")
        return coord_orig, feat_orig, labels_orig


class SemSegToothDataset(Dataset):
    def __init__(self, root='/data/3D_tooth_seg/3D_tooth', npoints=120000, split='train', shuffle=False, n_class=2, f_cols=8, b_limit_point=True):
        self.npoints = npoints
        self.root = os.path.join(root, split)
        self.f_cols = f_cols
        self.b_limit_point = b_limit_point
        self.mode = split
        print("load data from :", self.root)
        npy_files = glob.glob(os.path.join(self.root, "*.npy"))
        # print("all_data: ", len(npy_files))
        # rot = ["X15", "X30", "X45", "X60", "X75", "X90", "Y15", "Y30", "Y45", "Y60", "Y75", "Y90"]
        # npy_files = [file for file in npy_files if file[-13:-10] not in rot]
        n_data = len(npy_files)
        new_files = []
        print("ori_data: ", len(npy_files))
        if split != "test":
            label_weights = np.zeros(n_class)
            for idx, npy_file in enumerate(npy_files):
                # print(idx, npy_file)
                data = np.load(npy_file).astype(np.float32) 
                # data = data[data[:, 5] < 0]  # nz < 0
                # if len(data) > npoints:
                #     continue
                # else:
                #     new_files.append(npy_file)
                new_files.append(npy_file)
                # # print(npy_file, data.shape)
                labels = data[:, -1].astype(np.int32)
                # labels[labels < 1] = 0
                # labels[labels >= 1] = 1
                 
                # labels = data[:, -1].astype(np.int32)  # 鑾峰彇鏍囩鍒?                # # print(len(labels[labels > 0]))
                # m_labels = data[:, -2].astype(np.int32)
                # labels[(m_labels == 1) & (labels > 0)] = 1
                # labels[(m_labels == 2) & (labels > 0)] = 2
                # labels[(m_labels == 3) & (labels > 0)] = 3 
                tmp, _ = np.histogram(labels, range(n_class + 1))
                label_weights += tmp
            print("label_weights: ", label_weights)
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = np.amax(label_weights) / label_weights
            print("label_weights: ", self.label_weights)
        npy_files = new_files
        print("filter_data: ", len(npy_files))
        if shuffle:
            random.shuffle(npy_files)  # 闅忔満鎵撲贡 

        self.data_path = npy_files

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'toothModel': [0, 1]}

    def __getitem__(self, index):
        fn = self.data_path[index] 
        data = np.load(fn).astype(np.float32)
        # print(data.shape)
        # data = data[data[:, 5] < 0]  # remove nz > 0
        # print("---", data.shape)
        # np.random.shuffle(data)
        if self.mode == 'train':
            np.random.shuffle(data)
        if self.b_limit_point:
            data = data[:self.npoints, :]
        # print(fn, data.shape)
        point_set = data[:, 0:self.f_cols]

        labels = data[:, -1].astype(np.int32)
        
        coord = pc_normalize(point_set[:, 0:3]) 
        feat  = pc_normalize(point_set[:, 3:self.f_cols])  

        if self.mode == 'train':
            original_data = (coord, feat, labels)

            # if np.random.choice([0, 1]):
            #     coord, feat, labels = safe_augmentation(original_data, rotate_point_cloud_z_with_normal)

            # if np.random.choice([0, 1]):
            #     random_shift = RandomShift()
            #     coord, feat, labels = safe_augmentation((coord, feat, labels), random_shift)

            # if np.random.choice([0, 1]):
            #     random_scale = RandomScale()
            #     coord, feat, labels = safe_augmentation((coord, feat, labels), random_scale)

            # if np.random.choice([0, 1]):
            #     random_rotate = RandomRotate()
            #     coord, feat, labels = safe_augmentation((coord, feat, labels), random_rotate)

            augmentation_options = [
                # ("rotate_z", rotate_point_cloud_z_with_normal),
                # ("shift", lambda c, f, l: RandomShift()(c, f, l)),
                # ("scale", lambda c, f, l: RandomScale()(c, f, l)),
                # ("rotate", lambda c, f, l: RandomRotate()(c, f, l)),
                ("none", None) 
            ]
            
            aug_name, aug_method = random.choice(augmentation_options)
            if aug_method is not None:  
                coord, feat, labels = safe_augmentation(original_data, aug_method)

        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = torch.FloatTensor(coord)
        feat = torch.FloatTensor(feat)
        label = torch.LongTensor(labels) 
        return coord, feat, label

    def __len__(self):
        return len(self.data_path)



class ClsToothDataset(Dataset):
    def __init__(self, root='/data/3D_tooth_seg/3D_tooth', npoints=120000, split='train', shuffle=False, n_class=2, f_cols=8, b_limit_point=True):
        self.npoints = npoints
        if split != "all":
            self.root = os.path.join(root, split)
        else:
            self.root = root
        self.f_cols = f_cols
        self.b_limit_point = b_limit_point
        self.mode = split
        self.n_class = n_class
        print("load data from :", self.root)

        class_folders = sorted([d for d in os.listdir(self.root) 
                               if os.path.isdir(os.path.join(self.root, d))])
        if len(class_folders) != n_class:
            print(f"Warning: Finded {len(class_folders)} dirs, but set to {n_class} class!")
            self.n_class = len(class_folders)

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        self.data_files = []   
        self.data_labels = []   
        self.class_counts = np.zeros(self.n_class, dtype=np.int32)  
        print(f"鍙戠幇{self.n_class}涓被鍒? {list(self.class_to_idx.keys())}") 

        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(self.root, class_name)
            npy_files = glob.glob(os.path.join(class_path, "*.npy"))
            
            print(f"class {class_name}({class_idx}): {len(npy_files)} samples")
            
            for npy_file in npy_files:
                self.data_files.append(npy_file)
                self.data_labels.append(class_idx)
                self.class_counts[class_idx] += 1
        
        print(f"鎬诲叡鍔犺浇 {len(self.data_files)} 涓牱鏈?)
        print(f"鍚勭被鍒牱鏈暟: {self.class_counts}")
        
        if split != "test":
            label_weights = np.zeros(self.n_class)
            for idx, npy_file in enumerate(self.data_files):
                class_idx = self.data_labels[idx]
                label_weights[class_idx] += 1
            
            print("鍘熷绫诲埆缁熻: ", label_weights)
            
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)  
            self.label_weights = np.amax(label_weights) / label_weights  
            
            # self.label_weights = 1.0 / (label_weights + 1e-6)
            # self.label_weights = self.label_weights / np.sum(self.label_weights)
            
            print("绫诲埆鏉冮噸: ", self.label_weights)
        else:
            self.label_weights = np.ones(self.n_class)

        if shuffle:
            indices = list(range(len(self.data_files)))
            random.shuffle(indices)
            self.data_files = [self.data_files[i] for i in indices]
            self.data_labels = [self.data_labels[i] for i in indices]
        
        self.seg_classes = {'toothModel': list(range(self.n_class))}
    
    def __len__(self):
        return len(self.data_files)  

    def __getitem__(self, index):
        fn = self.data_files[index] 
        data = np.load(fn).astype(np.float32)
        # print("*****", self.data_labels)
        class_label = self.data_labels[index]
        # print(data.shape)
        # data = data[data[:, 5] < 0]  # remove nz > 0
        # print("---", data.shape)
        # np.random.shuffle(data)
        if self.mode == 'train':
            np.random.shuffle(data)
        if self.b_limit_point and len(data) > self.npoints:
            data = data[:self.npoints, :]
        # elif len(data) < self.npoints:
        #     indices = np.random.choice(len(data), self.npoints, replace=True)
        #     data = data[indices]
        # print(fn, data.shape)
        point_set = data[:, 0:self.f_cols] 
        
        coord = pc_normalize(point_set[:, 0:3]) 
        feat  = pc_normalize(point_set[:, 3:self.f_cols])  

        if self.mode == 'train':
            original_data = (coord, feat, class_label)

            # if np.random.choice([0, 1]):
            #     coord, feat, labels = safe_augmentation(original_data, rotate_point_cloud_z_with_normal)

            # if np.random.choice([0, 1]):
            #     random_shift = RandomShift()
            #     coord, feat, labels = safe_augmentation((coord, feat, labels), random_shift)

            # if np.random.choice([0, 1]):
            #     random_scale = RandomScale()
            #     coord, feat, labels = safe_augmentation((coord, feat, labels), random_scale)

            # if np.random.choice([0, 1]):
            #     random_rotate = RandomRotate()
            #     coord, feat, labels = safe_augmentation((coord, feat, labels), random_rotate)

            augmentation_options = [
                # ("rotate_z", rotate_point_cloud_z_with_normal),
                # ("shift", lambda c, f, l: RandomShift()(c, f, l)),
                # ("scale", lambda c, f, l: RandomScale()(c, f, l)),
                # ("rotate", lambda c, f, l: RandomRotate()(c, f, l)),
                ("none", None) 
            ]
            
            aug_name, aug_method = random.choice(augmentation_options)
            if aug_method is not None:  
                coord, feat, class_label = safe_augmentation(original_data, aug_method)

        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = torch.FloatTensor(coord)
        feat = torch.FloatTensor(feat)
        # print("-----------", class_label)
        label = torch.tensor(class_label, dtype=torch.long)
        # print("&&&&&&&&&&&&&&&&&&&&label: ", label) 
        return coord, feat, label



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


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


 

if __name__ == "__main__":
    # val_data = SemSegToothDataset(split='val', root="/data/3D_tooth_seg/3D_tooth/", npoints=40000, n_class=2, f_cols=8, b_limit_point=False)  
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, sampler=None, collate_fn=collate_fn)
    # for i, (coord, feat, target, offset) in enumerate(val_loader):
    #     if i > 0:
    #         continue
    #     print(coord.shape, feat.shape, target.shape, offset)
    train_data = SemSegToothDataset(split='train', root="/data/3D_tooth_seg/3D_tooth/", npoints=40000, n_class=2, f_cols=8, b_limit_point=False)  
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, sampler=None, collate_fn=collate_fn)
    for i, (coord, feat, target, offset) in enumerate(train_loader):
        if i > 0:
            continue
        print(coord.shape, feat.shape, target.shape, offset)
