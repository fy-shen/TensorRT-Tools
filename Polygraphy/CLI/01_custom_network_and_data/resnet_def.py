
from polygraphy import mod
from polygraphy import func
from polygraphy.backend.trt import NetworkFromOnnxPath
trt = mod.lazy_import('tensorrt')

parse_network_from_onnx = NetworkFromOnnxPath('../../models/ResNet-18.onnx')


@func.extend(parse_network_from_onnx)
def load_network(builder, network, parser):
    print(f'Network old name: {network.name}')
    network.name = 'ResNet-18'
    print(f'Network new name: {network.name}')
