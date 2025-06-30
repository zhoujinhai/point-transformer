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
import onnxruntime as ort
from onnxruntime_extensions import PyOp, onnx_op, get_library_path
import pointops_cuda

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
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro_export as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(in_channels=args.fea_dim, n_cls=args.classes).cuda()
    
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
    export(model)


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
    # np.random.shuffle(data)
    ori_neg_data = data[:, :3]

    point_set = data[:, 0:f_cols]
    # np.savetxt("./feats.txt", point_set)
    coord = pc_normalize(point_set[:, 0:3]) 
    feat  = pc_normalize(point_set[:, 3:f_cols])  
    # print("after_pc_normalize:", point_set.shape, label.shape)

    coord_min = np.min(coord, 0)
    coord -= coord_min
    
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) 
    
    return torch.cat([coord, feat], dim=1), ori_normal_data, ori_neg_data

def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    feat = feat / 255.
    return coord, feat

def show_pcl_data(data, label_cls=-1):
    import vedo
    points = data[:, 0:3] 

    colours = ["grey", "red", "blue", "yellow", "brown", "green", "black", "pink"]
    labels = data[:, label_cls]  # 最后一列为标签列
    diff_label = np.unique(labels)
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
        pt = vedo.Points(point.reshape(-1, 3)).c((colours[int(group_labels[i]) % len(colours)]))  # 显示点
        show_pts.append(pt)
    vedo.show(show_pts)


def export(model): 
    args.batch_size_test = 1
    model.eval()
     
    # test
    file_path = "/data/support/consurmer/test/missing/stl_pt_sample_aug/1e087f16-b0ce-46bc-9b11-7038ffd13013_Y30_point.npy"
    file_path = "/data/support/0321/XX8V3_VS_SET_VSc1_Subsetup1_Maxillar_point.npy"
   
    feats, ori_normal_data, ori_neg_data = data_load(file_path, args.fea_dim)
    print(feats.shape)
   
    feats = feats.cuda()
    with torch.no_grad():
        pred_part = model(feats)  # (n, k)
        print("pred_part: ", pred_part, pred_part.shape)
        # probs = torch.exp(pred_part)
        # print(probs.shape)
        # max_values, max_indices = torch.max(probs, dim=1)
        # print(max_values, max_values.shape, max_indices, max_indices.shape)
        # pred_labels = max_indices.cpu().numpy()
        # # nms_3d_point_cloud(ori_normal_neg_data[:, :3], labels, max_values[0].cpu().numpy())
        # show_data = np.c_[ori_neg_data, pred_labels]
        # print(show_data.shape, ori_normal_data.shape)
        # show_data = np.vstack((show_data, ori_normal_data))
        # print(show_data, show_data.shape)
        # # show_pcl_data(show_data)
    # exit()
    # export
    onnx_path = "./support.onnx"
    print("start convert model to onnx >>>")
    model.eval()
    print("********", feats.shape)
    torch.onnx.export(model,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (feats,),  # feats is (n, k)
                      onnx_path,
                      verbose=False,
                      input_names=["inputs"],
                      output_names=["res"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, # torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      dynamic_axes={
                          "inputs": {0: "n", 1: "c"},
                          "res": {0: "n", 1: "cls"}
                      }
                      )
    
    print("onnx model has exported!")

@onnx_op(
        op_type="FurthestSampling",                                     # 必须与 symbolic 中的 OpName 一致
        domain="ai.onnx.contrib",                                      # 必须与 symbolic 中的 domain 一致
        inputs=[PyOp.dt_float, PyOp.dt_int32, PyOp.dt_int32],           # 输入类型
        outputs=[PyOp.dt_int32],                                        # 输出类型
        since_version=12 
    )
def FurthestSampling(xyz, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        print("[DEBUG] PyOp called! Output shape:", xyz.shape)
        xyz = torch.from_numpy(xyz).cuda()
        offset = torch.from_numpy(offset).cuda()
        new_offset = torch.from_numpy(new_offset).cuda()
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i-1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        
        n_int = n.item() if isinstance(n, torch.Tensor) else n  # 确保 n 为标量
        tmp = torch.full((n_int,), 1e10, dtype=torch.float32, device='cuda')
        
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        # idx = np.random.choice(xyz.shape[0], new_offset[-1], replace=False).astype(np.int32)
        return idx


# @onnx_op(
#         op_type="FurthestSampling",                                     # 必须与 symbolic 中的 OpName 一致
#         domain="ai.onnx.contrib",                                      # 必须与 symbolic 中的 domain 一致
#         inputs=[PyOp.dt_float, PyOp.dt_int32, PyOp.dt_int32],           # 输入类型
#         outputs=[PyOp.dt_int32],                                        # 输出类型
#         since_version=12 
#     )
# def FurthestSampling(xyz, offset, new_offset):
#         """
#         input: xyz: (n, 3), offset: (b), new_offset: (b)
#         output: idx: (m)
#         """
#         print("[DEBUG] PyOp called! Output shape:", xyz.shape)
#         xyz = torch.from_numpy(xyz)
#         offset = torch.from_numpy(offset)
#         new_offset = torch.from_numpy(new_offset)
         
#         torch.ops.load_library("/home/heygears/jinhai_zhou/learn/point-transformer/tool/inference_C++/libFurthestSampling.so")
#         fps = torch.ops.my_ops.FurthestSampling 
#         idx = fps(xyz, offset, new_offset)
#         return idx


@onnx_op(
        op_type="KNNQuery",                                                                           # 必须与 symbolic 中的 OpName 一致
        domain="ai.onnx.contrib",                                                                     # 必须与 symbolic 中的 domain 一致
        inputs=[PyOp.dt_int32, PyOp.dt_float, PyOp.dt_float, PyOp.dt_int32, PyOp.dt_int32],           # 输入类型
        outputs=[PyOp.dt_int32, PyOp.dt_float],                                                       # 输出类型   idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample) _is not used! so, the out just one
        since_version=12 
    )
def KNNQuery(nsample, xyz, new_xyz, offset, new_offset):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
    output: idx: (m, nsample), dist2: (m, nsample)
    """
    
    nsample = torch.from_numpy(nsample).cuda()
    xyz = torch.from_numpy(xyz).cuda()
    new_xyz = torch.from_numpy(new_xyz).cuda() 
    offset = torch.from_numpy(offset).cuda()
    new_offset = torch.from_numpy(new_offset).cuda()

    if new_xyz is None: new_xyz = xyz
    assert xyz.is_contiguous() and new_xyz.is_contiguous()
    m = new_xyz.shape[0]
    idx = torch.cuda.IntTensor(m, nsample).zero_()
    dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
    pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
    return idx,  torch.sqrt(dist2)


# @onnx_op(
#         op_type="KNNQuery",                                                                           # 必须与 symbolic 中的 OpName 一致
#         domain="ai.onnx.contrib",                                                                     # 必须与 symbolic 中的 domain 一致
#         inputs=[PyOp.dt_int32, PyOp.dt_float, PyOp.dt_float, PyOp.dt_int32, PyOp.dt_int32],           # 输入类型
#         outputs=[PyOp.dt_int32],                                                                      # 输出类型   idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample) _is not used! so, the out just one
#         since_version=12 
#     )
# def KNNQuery(nsample, xyz, new_xyz, offset, new_offset):
#     """
#     input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
#     output: idx: (m, nsample), dist2: (m, nsample)
#     """
    
#     nsample = torch.from_numpy(nsample)
#     xyz = torch.from_numpy(xyz)
#     new_xyz = torch.from_numpy(new_xyz)
#     offset = torch.from_numpy(offset)
#     new_offset = torch.from_numpy(new_offset)

#     torch.ops.load_library("/home/heygears/jinhai_zhou/learn/point-transformer/tool/inference_C++/libKNNQuery.so")
#     knn_query = torch.ops.my_ops.KNNQuery
#     print("nsample: ", nsample, "xyz: ", xyz.shape, "new_xyz: ", new_xyz.shape, "offset: ", offset, "new_offset: ", new_offset)
#     idx, dist2 = knn_query(nsample, xyz, new_xyz, offset, new_offset) 
#     print("idx: ", idx.shape)
#     return idx, dist2

def inference():   
    
    session_options = ort.SessionOptions() 
    session_options.register_custom_ops_library(get_library_path()) 
    session_options.log_severity_level = 2
    sess = ort.InferenceSession("./support.onnx", providers=["CPUExecutionProvider"], sess_options=session_options)
    # test
    file_path = "/data/support/0321/XX8V3_VS_SET_VSc1_Subsetup1_Maxillar_point.npy"
    file_path = "/data/support/consurmer/test/missing/stl_pt_sample_aug/1e087f16-b0ce-46bc-9b11-7038ffd13013_Y30_point.npy"
    feats, ori_normal_data, ori_neg_data = data_load(file_path, 12)
    feats = feats.numpy()
    print(feats.shape) 
    
    outputs = sess.run(
        output_names=["res"],
        input_feed = {"inputs": feats}
    )[0]

    print("outputs: ", outputs.shape)
    print("output data: ", outputs)  

if __name__ == '__main__':
    main()
    inference()
