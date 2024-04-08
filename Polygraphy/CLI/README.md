# Polygraphy CLI

## 简介

Polygraphy CLI 内包含如下多个子工具
1. run：在不同后端（如 TensorRT 和 Onnx-Runtime）下运行推理和比较结果，
主要用于测试精度，找到精度不足的层
2. convert：模型转换，如 onnx -> trt
3. inspect：查看信息，如网络结构、参数、数据、TensorRT 优化策略等
4. check：检查和验证模型的各个方面
5. surgeon：修改优化 onnx 模型
6. template：（实验阶段）生成脚本模板
7. debug：（实验阶段）调试各种模型问题
8. data：操作其他 Polygraphy 子工具生成的输入输出数据


## 示例简介

### 0. [基础示例](BaseExamples.md)
此例为 Polygraphy CLI 的基础用法，包括对比模型在 TensorRT 和 Onnx-Runtime 下的推理精度、
保存和加载推理的输入输出、生成 Python 脚本、检查网络推理过程是否出现异常值。

### 1. [自定义网络和数据](01_custom_network_and_data%2FREADME.md)
此例在 ResNet-18 末尾添加一个 Softmax，并使用真实图像作为输入，与 Pytorch 的输出对比精度。

### 2. [自定义精度](02_precision_constraints%2FREADME.md)
此例用 onnx-graphsurgeon 创建一个网络，手动指定各层精度，强制触发 FP16 精度溢出问题。

### 3. [提取子图](03_subgraph%2FREADME.md)
此例提取 ResNet-18 一部分作为子图；
人工创建一个不能转 TensorRT 的网络，分割可以转和不能转的子图。