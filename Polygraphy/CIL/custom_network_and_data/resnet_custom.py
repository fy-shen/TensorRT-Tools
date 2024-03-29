from polygraphy import mod
from polygraphy import func
from polygraphy.backend.trt import NetworkFromOnnxPath
from polygraphy.backend.trt import CreateConfig

trt = mod.lazy_import('tensorrt')

resnet_onnx = NetworkFromOnnxPath('../../models/ResNet-18.onnx')


@func.extend(CreateConfig())
def custom_config(builder, network, config):
    config.set_flag(trt.BuilderFlag.FP16)


@func.extend(resnet_onnx)
def custom_network(builder, network, parser):
    prev_output = network.get_output(0)
    network.unmark_output(prev_output)

    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Network.html?highlight=add_softmax#/tensorrt.INetworkDefinition.add_softmax
    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Layers.html#/isoftmaxlayer
    # https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/SoftMax.html#/
    softmax_layer = network.add_softmax(prev_output)
    softmax_layer.axes = 1 << 1
    softmax_layer.name = 'softmax'
    output = softmax_layer.get_output(0)
    output.name = 'softmax_output_0'
    network.mark_output(output)
