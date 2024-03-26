
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from common import get_onnx_path

if __name__ == '__main__':
    onnx_file = get_onnx_path('ResNet_div_zero.onnx')

    graph = gs.import_onnx(onnx.load('../models/ResNet-18.onnx'))

    # 修改最后一个节点的输出，与整体命名统一
    fc_node = graph.nodes[-1]
    fc_out = gs.Variable('/fc/Gemm_output_0', dtype=np.float32)
    fc_node.outputs = [fc_out]

    div_value = np.zeros((1000,), dtype=np.float32)
    div_const = gs.Constant(name='onnx::div', values=div_value)
    div_out = gs.Variable('y', dtype=np.float32, shape=['batch', 1000])
    div_node = gs.Node(op='Div', inputs=[fc_node.outputs[0], div_const], outputs=[div_out])

    graph.nodes.append(div_node)
    graph.outputs = [div_out]

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), onnx_file)
