
import torch
import torchvision.models as models
from common import get_onnx_path


if __name__ == '__main__':
    onnx_file = get_onnx_path('ResNet-18.onnx')

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights, progress=False).eval().cuda()

    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224, device='cuda'),
        onnx_file,
        input_names=['x'],
        output_names=['y'],
        do_constant_folding=True,
        verbose=True,
        opset_version=12,
        dynamic_axes={'x': {0: 'batch'}, 'y': {0: 'batch'}},
    )
