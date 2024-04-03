from polygraphy import func
from polygraphy.backend.trt import NetworkFromOnnxPath
import tensorrt as trt


parse_network_from_onnx = NetworkFromOnnxPath('./needs_constraints.onnx')


@func.extend(parse_network_from_onnx)
def load_network(builder, network, parser):
    for layer in network:
        if layer.name == 'Add':
            layer.precision = trt.float16
            layer.set_output_type(0, trt.float16)

        if layer.name == 'Sub':
            layer.precision = trt.float32
            layer.set_output_type(0, trt.float32)


@func.extend(parse_network_from_onnx)
def load_network_fp16(builder, network, parser):
    for layer in network:
        if layer.name in ('Add', 'Sub'):
            layer.precision = trt.float16
            layer.set_output_type(0, trt.float16)
