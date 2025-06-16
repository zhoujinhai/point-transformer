import random
import numpy as np 
import torch
import torch.nn as nn 

import sys 
sys.path.append("/home/heygears/jinhai_zhou/learn/point-transformer/")
 
from lib.pointops.functions import pointops
import pointops_cuda
import onnxruntime as ort
from onnxruntime_extensions import PyOp, onnx_op, get_library_path

random.seed(123)
np.random.seed(123)

class QueryAndGroup(nn.Module):
    def __init__(self, stride=400, nsample=8):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        
    def forward(self, feats):
        p = feats[:, :3]
        x = feats[:, 3:]
        p = p.contiguous()
        x = x.contiguous()
        o = torch.IntTensor([feats.shape[0]]).to(p.device)  # (n, 3), (n, c), (b)
         
        n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
        for i in range(1, o.shape[0]):
            count += (o[i].item() - o[i-1].item()) // self.stride
            n_o.append(count)
        n_o = torch.cuda.IntTensor(n_o) 
        # idx = pointops.furthestsampling(p, o, n_o)  # (m)
        # print(idx, idx.shape)
        
        idx_base = torch.arange(p.shape[0] + 1, device='cuda', dtype=torch.int32)
        idx = idx_base[:n_o[0]] 
        # print(idx, idx.shape)
        n_p = p[idx.long(), :]  # (m, 3)
        x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
        return x  
     

def main():  
    model = QueryAndGroup().cuda() 
    
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
    onnx_path = "./QueryAndGroup.onnx"
    print("start convert model to onnx >>>")
    model.eval()
    print("********", feats.shape)
    torch.onnx.export(model,  # support torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
                      (feats,),  # feats is (n, k)
                      onnx_path,
                      verbose=False,
                      input_names=["feats"],
                      output_names=["outputs"],
                      opset_version=12,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, # torch.onnx.OperatorExportTypes.ONNX,  # ONNX_ATEN_FALLBACK,
                      # custom_opsets={"PointCloudDomain": 12},  # 域名和版本需与 PyOp 一致
                      dynamic_axes={
                          "feats": {0: "n", 1: "c"},
                          "outputs": {0: "n", 1: "m", 2: "c"}
                      }
                      )
    
    print("onnx QueryAndGroup model has exported!")


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
    
#     nsample = torch.from_numpy(nsample).cuda()
#     xyz = torch.from_numpy(xyz).cuda()
#     new_xyz = torch.from_numpy(new_xyz).cuda() 
#     offset = torch.from_numpy(offset).cuda()
#     new_offset = torch.from_numpy(new_offset).cuda()

#     if new_xyz is None: new_xyz = xyz
#     assert xyz.is_contiguous() and new_xyz.is_contiguous()
#     m = new_xyz.shape[0]
#     idx = torch.cuda.IntTensor(m, nsample).zero_()
#     dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
#     pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
#     return idx, torch.sqrt(dist2)

@onnx_op(
        op_type="KNNQuery",                                                                           # 必须与 symbolic 中的 OpName 一致
        domain="ai.onnx.contrib",                                                                     # 必须与 symbolic 中的 domain 一致
        inputs=[PyOp.dt_int32, PyOp.dt_float, PyOp.dt_float, PyOp.dt_int32, PyOp.dt_int32],           # 输入类型
        outputs=[PyOp.dt_int32],                                                                      # 输出类型   idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample) _is not used! so, the out just one
        since_version=12 
    )
def KNNQuery(nsample, xyz, new_xyz, offset, new_offset):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
    output: idx: (m, nsample), dist2: (m, nsample)
    """
    
    nsample = torch.from_numpy(nsample)
    xyz = torch.from_numpy(xyz)
    new_xyz = torch.from_numpy(new_xyz)
    offset = torch.from_numpy(offset)
    new_offset = torch.from_numpy(new_offset)

    torch.ops.load_library("/home/heygears/jinhai_zhou/learn/point-transformer/tool/inference_C++/libKNNQuery.so")
    knn_query = torch.ops.my_ops.KNNQuery
    print("nsample: ", nsample, "xyz: ", xyz.shape, "new_xyz: ", new_xyz.shape, "offset: ", offset, "new_offset: ", new_offset)
    idx, dist2 = knn_query(nsample, xyz, new_xyz, offset, new_offset) 
    print("idx: ", idx.shape)
    return idx, dist2


def inference():    
    session_options = ort.SessionOptions() 
    session_options.register_custom_ops_library(get_library_path())
    sess = ort.InferenceSession("./QueryAndGroup.onnx", providers=["CPUExecutionProvider"], sess_options=session_options)
    # test
    file_path = "/data/support/0321/XX8V3_VS_SET_VSc1_Subsetup1_Maxillar_point.npy"
   
    feats, ori_normal_data, ori_neg_data = data_load(file_path, 12)
    feats = feats.numpy()
    print(feats.shape) 
    
    outputs = sess.run(
        output_names=["outputs"],
        input_feed = {"feats": feats}
    )[0]

    print("outputs: ", outputs.shape)
    print("output data: ", outputs)  


if __name__ == '__main__':
    main()
    inference() 
