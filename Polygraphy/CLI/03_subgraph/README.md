
# 提取子图

首先可以查看模型结构，确定需要提取子图的输入输出张量名称
```shell
polygraphy inspect model ../../models/ResNet-18.onnx --show layers
```

当张量的形状和数据类型未知时，可以使用 `auto` 让 Polygraphy 自动推导。
输入需要形状和数据类型，输出只需要数据类型。
```shell
polygraphy surgeon extract ../../models/ResNet-18.onnx \
    --inputs /layer3/layer3.1/relu_1/Relu_output_0:auto:auto \
    --outputs y:auto -o subgraph.onnx
```
形状和数据类型已知时，也可以按如下方式指定。
```shell
polygraphy surgeon extract ../../models/ResNet-18.onnx \
    --inputs /layer3/layer3.1/relu_1/Relu_output_0:[batch,256,14,14]:float32 \
    --outputs y:float32 -o subgraph.onnx
```
查看子图结构
```shell
polygraphy inspect model subgraph.onnx \
    --show layers
```

注：

当 `auto` 指定了形状或数据类型时，Polygraphy 依赖 ONNX shape inference 来推导中间张量的形状和数据类型。

如果 ONNX shape inference 无法确定形状，Polygraphy 会使用 ONNX-Runtime 和合成数据推理模型，
可以使用 `--model-inputs` 和 `Data Loader` 下选项设置输入数据的形状和内容。

这样会导致生成的子图输入为固定的形状，可以在子图上再次使用 `extract` 并设置动态尺寸的形状。


# 分割能否转 TensorRT 的子图
创建一个网络，其中包含一个 TensorRT 不支持的 `NonZero` 节点
```shell
python create_network.py
```
分析 Onnx 转 TensorRT 失败原因，并将可以转换和不能转换的子图分割保存
```shell
polygraphy inspect capability model.onnx
```
