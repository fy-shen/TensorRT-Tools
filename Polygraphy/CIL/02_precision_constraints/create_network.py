
import numpy as np
import onnx
import onnx_graphsurgeon as gs


x = gs.Variable(name='x', dtype=np.float32, shape=(1, 1, 256, 256))
y = gs.Variable(name='y', dtype=np.float32, shape=(1, 1, 256, 256))

rot90_value = np.fliplr(np.eye(256, dtype=np.float32)).reshape(1, 1, 256, 256)
rot90 = gs.Constant(name='rot90', values=rot90_value)
fp16_max_value = np.array(65504, dtype=np.float32).reshape(1, 1, 1, 1)
fp16_max = gs.Constant(name='fp16_max', values=fp16_max_value)

matmul_out_0 = gs.Variable(name='matmul_out_0')
add_out_1 = gs.Variable(name='add_out_1')
sub_out_2 = gs.Variable(name='sub_out_2')

nodes = [
    gs.Node(op='MatMul', name='MatMul_0', inputs=[x, rot90], outputs=[matmul_out_0]),
    gs.Node(op='Add', name='Add', inputs=[matmul_out_0, fp16_max], outputs=[add_out_1]),
    gs.Node(op='Sub', name='Sub', inputs=[add_out_1, fp16_max], outputs=[sub_out_2]),
    gs.Node(op='MatMul', name='MatMul_1', inputs=[sub_out_2, rot90], outputs=[y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x], outputs=[y])
onnx.save(gs.export_onnx(graph), 'needs_constraints.onnx')

