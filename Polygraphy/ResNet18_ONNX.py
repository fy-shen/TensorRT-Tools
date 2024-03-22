import torch
import torchvision.models as models


if __name__ == '__main__':
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights, progress=False).eval().cuda()

    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224, device='cuda'),
        'models/ResNet-18.onnx',
        input_names=['x'],
        output_names=['y'],
        do_constant_folding=True,
        verbose=True,
        opset_version=12,
        dynamic_axes={'x': {0: 'batch'}, 'y': {0: 'batch'}},
    )
