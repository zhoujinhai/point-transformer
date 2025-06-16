import random
import numpy as np

from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Function

import sys 
sys.path.append("/home/heygears/jinhai_zhou/learn/point-transformer/")
 
from util.voxelize import voxelize
from lib.pointops.functions import pointops
import pointops_cuda
import onnxruntime as ort
from onnxruntime_extensions import PyOp, onnx_op,  get_library_path

random.seed(123)
np.random.seed(123)

class FPS(nn.Module):
    def __init__(self, stride=400, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        
    def forward(self, feats):
        p = feats[:, :3]
        x = feats[:, 3:]
        p = p.contiguous()
        x = x.contiguous()
        o = torch.IntTensor([feats.shape[0]]).to(p.device)  # (n, 3), (n, c), (b)
        print("p0", p.shape, x.shape, o, o.shape)
        # p, x, o = pxo  # (n, 3), (n, c), (b)
        
        n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
        for i in range(1, o.shape[0]):
            count += (o[i].item() - o[i-1].item()) // self.stride
            n_o.append(count)
        n_o = torch.cuda.IntTensor(n_o)
        # print(p, o, n_o)
        idx = pointops.furthestsampling(p, o, n_o)  # (m)
      
        return idx  
    

def main():
    model = FPS().cuda()
    
    export(model)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def data_load(data_name, f_cols, b_shuffle = False):
    data = np.load(data_name).astype(np.float32) 
    ori_normal_data = data[data[:, 5] >= 0][:, :4]  
    ori_normal_data[:, -1] = 0  
    # print(data.shape)
    data = data[data[:, 5] < 0]  # remove nz > 0
    # print("---", data.shape)
    if b_shuffle:
        np.random.shuffle(data)
    ori_neg_data = data[:, :3]

    point_set = data[:, 0:f_cols]
 
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

def export(model):  
    model.eval()
     
    # test
    file_path = "/data/support/0321/XX8V3_VS_SET_VSc1_Subsetup1_Maxillar_point.npy"
   
    feats, ori_normal_data, ori_neg_data = data_load(file_path, 12)
    print(feats.shape)
    feats = feats.cuda()
    with torch.no_grad():
        pred_part = model(feats)  # (n, k)
        print("pred_part: ", pred_part, pred_part.shape)  
    
    # export
    onnx_path = "./fps.onnx"
    print("start convert model to onnx >>>")
    model.eval()
    print("********", feats.shape)
    torch.onnx.export(model,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (feats,),  # feats is (n, k)
                      onnx_path,
                      verbose=False,
                      input_names=["feats"],
                      output_names=["index"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, # torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      # custom_opsets={"PointCloudDomain": 12},  # 域名和版本需与 PyOp 一致
                      dynamic_axes={
                          "feats": {0: "n", 1: "c"},
                          "index": {0: "n"}
                      }
                      )
    
    print("onnx fps model has exported!")

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
#         xyz = torch.from_numpy(xyz).cuda()
#         offset = torch.from_numpy(offset).cuda()
#         new_offset = torch.from_numpy(new_offset).cuda()
#         assert xyz.is_contiguous()
#         n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
#         for i in range(1, b):
#             n_max = max(offset[i] - offset[i-1], n_max)
#         idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        
#          n_int = n.item() if isinstance(n, torch.Tensor) else n  # 确保 n 为标量
#         tmp = torch.full((n_int,), 1e10, dtype=torch.float32, device='cuda')
        
#         pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
#         del tmp
#         # idx = np.random.choice(xyz.shape[0], new_offset[-1], replace=False).astype(np.int32)
#         return idx


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
        xyz = torch.from_numpy(xyz)
        offset = torch.from_numpy(offset)
        new_offset = torch.from_numpy(new_offset)
         
        torch.ops.load_library("/home/heygears/jinhai_zhou/learn/point-transformer/tool/inference_C++/libFurthestSampling.so")
        fps = torch.ops.my_ops.FurthestSampling 
        idx = fps(xyz, offset, new_offset)
        return idx

def inference():    
    session_options = ort.SessionOptions() 
    session_options.register_custom_ops_library(get_library_path())
    sess = ort.InferenceSession("./fps.onnx", providers=["CPUExecutionProvider"], sess_options=session_options)
    # test
    file_path = "/data/support/0321/XX8V3_VS_SET_VSc1_Subsetup1_Maxillar_point.npy"
   
    feats, ori_normal_data, ori_neg_data = data_load(file_path, 12)
    feats = feats.numpy()
    np.savetxt("./feats.txt", feats)
    print(feats.shape) 
    
    outputs = sess.run(
        output_names=["index"],
        input_feed = {"feats": feats}
    )[0]

    print("outputs: ", outputs, outputs.shape)  

 
if __name__ == '__main__':
    main()
    inference() 

   
