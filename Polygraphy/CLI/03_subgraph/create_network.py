import numpy as np
import onnx
import onnx_graphsurgeon as gs


x = gs.Variable(name='x', dtype=np.float32, shape=['Batch'])
y = gs.Variable(name='y', dtype=np.float32)
identity_out_0 = gs.Variable(name='identity_out_0', dtype=np.float32)
nonzero_out_1 = gs.Variable(name='nonzero_out_1', dtype=np.float32)


nodes = [
    gs.Node(op='Identity', inputs=[x], outputs=[identity_out_0]),
    gs.Node(op='NonZero', inputs=[identity_out_0], outputs=[nonzero_out_1]),
    gs.Node(op='Identity', inputs=[nonzero_out_1], outputs=[y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x], outputs=[y])
onnx.save(gs.export_onnx(graph), 'model.onnx')
