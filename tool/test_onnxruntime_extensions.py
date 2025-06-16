# ref: https://github.com/microsoft/onnxruntime-extensions/blob/main/test/test_pyops.py
import os
import onnx
import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef, make_onnx_model,
    get_library_path as _get_library_path)


def test_Inversion():
    import torch
    import torch.onnx 

    # 定义自定义算子（示例：矩阵逆运算 + 恒等连接）
    class InverseFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            inv_x = torch.inverse(x)
            return inv_x + x  # 自定义逻辑

        @staticmethod
        def symbolic(g, x):
            # 映射到 ONNX 自定义算子（域名::算子名）
            return g.op("ai.onnx.contrib::Inverse2", x)  #
        
    # 封装为模型
    class CustomModel(torch.nn.Module):
        def forward(self, x):
            return InverseFunction.apply(x)

    # 导出 ONNX 模型
    model = CustomModel()
    dummy_input = torch.randn(3, 3)  # 确保矩阵可逆
    torch.onnx.export(
        model, 
        dummy_input, 
        "custom_inverse.onnx",
        input_names=["input_matrix"],
        output_names=["output"],
        opset_version=12
    )
    
    import onnxruntime as ort
    import numpy as np 
 
    @onnx_op(op_type="Inverse2", domain="ai.onnx.contrib")
    def inverse2(x: np.ndarray):
        return np.linalg.inv(x) + x

    # 加载模型时传递 SessionOptions
    so = ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
        
    session = ort.InferenceSession("./custom_inverse.onnx", so, providers=['CPUExecutionProvider'])

    # 准备输入（确保矩阵可逆）
    input_matrix = np.array([
        [1.0, 0.5, 0.0],
        [0.2, 1.0, 0.3],
        [0.0, 0.1, 1.0]
    ], dtype=np.float32)

    # 运行推理
    output = session.run(
        output_names=["output"],
        input_feed={"input_matrix": input_matrix}
    )[0]

    print("自定义算子输出：\n", output)

    from onnxruntime_extensions import PyOrtFunction
    model_func = PyOrtFunction.from_model("./custom_inverse.onnx")
    out = model_func(input_matrix)
    print("out: ", out)
    

def _create_test_model():
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('PyReverseMatrix',
                                  ['identity1'], ['reversed'],
                                  domain='ai.onnx.contrib')]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.FLOAT, [None, 2])
    output0 = helper.make_tensor_value_info(
        'reversed', onnx_proto.TensorProto.FLOAT, [None, 2])

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = make_onnx_model(graph)
    # onnx.save(model, "test_PyReverseMatrix.onnx")
    return model


class TestPythonOp(unittest.TestCase):

    @classmethod
    def setUpClass(cls): 
        @onnx_op(op_type="PyReverseMatrix")
        def reverse_matrix(x):
            # The user custom op implementation here.
            return np.flip(x, axis=0).astype(np.float32)
        
     

    def test_python_operator(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model()
        self.assertIn('op_type: "PyReverseMatrix"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])
        # sess = _ort.InferenceSession("./test_PyReverseMatrix.onnx", so, providers=['CPUExecutionProvider'])
        # sess = _ort.InferenceSession("./custom_inverse.onnx", so, providers=['CPUExecutionProvider'])
        input_1 = np.array(
            [1, 2, 3, 4, 5, 6]).astype(np.float32).reshape([3, 2])
        txout = sess.run(None, {'input_1': input_1})
        assert_almost_equal(txout[0], np.array([[5., 6.], [3., 4.], [1., 2.]]))
        
  
if __name__ == "__main__":
    # unittest.main()
    test_Inversion()
