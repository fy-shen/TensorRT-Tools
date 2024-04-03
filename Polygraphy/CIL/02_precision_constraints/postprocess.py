import tensorrt as trt


def postprocess(network):
    for layer in network:
        if layer.name == 'Add':
            layer.precision = trt.float16
            layer.set_output_type(0, trt.float16)

        if layer.name == 'Sub':
            layer.precision = trt.float32
            layer.set_output_type(0, trt.float32)


def postprocess_fp16(network):
    for layer in network:
        if layer.name in ('Add', 'Sub'):
            layer.precision = trt.float16
            layer.set_output_type(0, trt.float16)
