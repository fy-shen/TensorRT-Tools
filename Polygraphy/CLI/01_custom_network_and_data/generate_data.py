import os
from PIL import Image
import numpy as np
from polygraphy.json import save_json
from polygraphy.comparator import RunResults

import torch
import torchvision.models as models


data_path = '../../data/images'
img_path = [os.path.join(data_path, img_name) for img_name in os.listdir(data_path)]
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def load_img(img_file):
    image = Image.open(img_file).resize((224, 224))
    image = (np.array(image, dtype=np.float32) / 255 - mean) / std
    image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)
    return image.astype(np.float32)


def load_data():
    for img_file in img_path:
        print(img_file)
        yield {'x': load_img(img_file)}


input_data = list(load_data())
save_json(input_data, 'custom_inputs.json', description='custom input data')

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights, progress=False).eval().cuda()
output_data = []
for data in input_data:
    output_tensor = model(torch.from_numpy(data['x']).cuda())
    output_data.append({'y': output_tensor.detach().cpu().numpy()})

outputs = RunResults()
outputs.add(output_data, runner_name='torch_runner')
outputs.save('torch_outputs.json')
