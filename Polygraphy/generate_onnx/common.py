import os
from os.path import abspath, dirname, join


def get_onnx_path(fn):
    onnx_dir = join(dirname(dirname(abspath(__file__))), 'models')
    os.makedirs(onnx_dir, exist_ok=True)
    return join(onnx_dir, fn)
