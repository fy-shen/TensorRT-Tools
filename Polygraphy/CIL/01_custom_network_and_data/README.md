

此例主要解决以下需求：
1. 用 TensorRT Python API 从头构建一个网络
2. 修改现有网络
3. 自定义 TensorRT 构建器配置
4. 使用自定义数据作为输入，而不是用 Polygraphy 生成的数据
5. 使用自定义数据作为输出对比，例如 Pytorch 的输出

这些可以通过 Polygraphy CIL 工具配合 Python 脚本实现。

## 生成模板

先生成模板，再编写脚本

```shell
polygraphy template trt-network -o create_network.py
polygraphy template trt-config -o trt_network_cfg.py
```
可以选择需要修改的网络；也可以用构建器配置选项填充脚本，例如启用 FP16
```shell
polygraphy template trt-network ../../models/ResNet-18.onnx -o create_network.py
polygraphy template trt-config --fp16 -o trt_network_cfg.py
```

## 示例一：Helloworld
修改模型的名称，并启用 FP16
```shell
polygraphy run --trt resnet_def.py --model-type=trt-network-script
```

```shell
polygraphy run --trt resnet_def.py --model-type=trt-network-script \
    --trt-config-script=resnet_cfg.py
```

## 示例二：修改网络和构建器配置
可以使用 `<file_name>.py:<function_name>` 的方式来代替 `load_network`、`load_config` 等函数

在 ResNet 末尾添加 `softmax`、启用 FP16 并推理
```shell
polygraphy run --trt resnet_custom.py:custom_network \
    --model-type=trt-network-script \
    --trt-config-script=resnet_custom.py:custom_config
```

用 `convert` 按类似的方式把模型转换为 TensorRT 引擎并保存
```shell
polygraphy convert resnet_custom.py:custom_network \
    -o resnet_fp32.engine
```
```shell
polygraphy convert resnet_custom.py:custom_network \
    --trt-config-script=resnet_custom.py:custom_config \
    -o resnet_fp16.engine
```
对比不同精度下的输出
```shell
polygraphy run resnet_fp32.engine --trt \
    --save-inputs input.json \
    --save-outputs output.josn
```
```shell
polygraphy run resnet_fp16.engine --trt \
    --load-inputs input.json \
    --load-outputs output.josn
```
用 `inspect model` 查看引擎的信息，`--show layers attrs weights` 可查看更多细节，


注：`--show layers` 仅在使用 `profiling_verbosity` 而不是 `None` 构建引擎时有效，详细程度越高，层信息越多。
```shell
polygraphy inspect model resnet_fp32.engine \
    --show layers
```
支持配合 `gerp` 筛选信息
```shell
polygraphy inspect model resnet_fp16.engine \
    --show layers | grep -i softmax
```
另外，可以将支持的格式转换为 TensorRT 网络并输出信息
```shell
polygraphy inspect model ../../models/ResNet-18.onnx \
    --show layers --display-as=trt
```

## 示例三：自定义输入输出
用 Python 脚本自定义输入数据，这里使用真实数据，读取图像并做预处理
```shell
polygraphy run ../../models/ResNet-18.onnx \
   --trt --onnxrt \
   --input-shapes x:[1,3,224,224] \
   --data-loader-script generate_data.py
```

可以将输入和 Pytorch 输出存储到本地
```shell
python generate_data.py
```
检查数据，`--show-values` 选项可显示数值
```shell
polygraphy inspect data custom_inputs.json
polygraphy inspect data torch_outputs.json
```

使用本地数据
```shell
polygraphy run ../../models/ResNet-18.onnx \
   --trt --onnxrt \
   --input-shapes x:[1,3,224,224] \
   --load-inputs custom_inputs.json
```

对比 TensorRT 和 Pytorch 输出
```shell
polygraphy run ../../models/ResNet-18.onnx \
   --trt \
   --input-shapes x:[1,3,224,224] \
   --load-inputs custom_inputs.json \
   --load-outputs torch_outputs.json
```

使用大量数据时，建议用脚本，避免数据写入磁盘。而 JSON 文件可能更便携，有助于确保可重复性。
