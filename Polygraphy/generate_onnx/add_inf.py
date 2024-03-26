
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from common import get_onnx_path

if __name__ == '__main__':
    onnx_file = get_onnx_path('add_inf.onnx')

    X = gs.Variable(name='X', dtype=np.float32, shape=[1])
    Y = gs.Variable(name='Y', dtype=np.float32, shape=[1])

    add_value = np.array([float('inf')], dtype=np.float32)
    add_const = gs.Constant(name='inf', values=add_value)
    add_node = gs.Node(op='Add', inputs=[X, add_const], outputs=[Y])

    graph = gs.Graph(nodes=[add_node], inputs=[X], outputs=[Y])
    onnx.save(gs.export_onnx(graph), onnx_file)
