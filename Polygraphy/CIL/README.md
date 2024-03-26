
# 1. 对比 TensorRT 和 ONNX-Runtime

```shell
polygraphy run ../models/ResNet-18.onnx \
    --trt --fp16 --onnxrt \
    --input-shapes x:[1,3,224,224]
```

由于模型输入是动态的，Polygraphy 会默认将动态维度的数值用 `constants.DEFAULT_SHAPE_VALUE`（=1）替代，并给出以下 warning。
通过 `--input-shapes` 可以指定输入形状。
```
[W] Input tensor: x (dtype=DataType.FLOAT, shape=(-1, 3, 224, 224)) | No shapes provided; Will use shape: [1, 3, 224, 224] for min/opt/max in profile.
[W] This will cause the tensor to have a static shape. If this is incorrect, please set the range of shapes for this input tensor.
```

TensorRT 启用 FP16 时给出了以下 warning
```
[W] TensorRT encountered issues when converting weights between types and that could affect accuracy.
[W] If this is not the desired behavior, please modify the weights or retrain with regularization to adjust the magnitude of the weights.
[W] Check verbose logs for the list of affected weights.
[W] - 23 weights are affected by this issue: Detected subnormal FP16 values.
[W] - 12 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.
```

默认误差（tolerance）通常适用于 FP32 精度，可以使用 `--atol` 和 `--rtol` 分别设置绝对误差和相对误差。
误差计算如下：
```
absdiff = abs(out0 - out1)
reldiff = absdiff / abs(out1)
```

`--check-error-stat` 可以更改对比的指标（例如 `mean` `median` `max`），默认为 `--check-error-stat elemwise` 即逐元素对比（elementwise）。

经测试，当绝对误差和相对误差都大于设定值时才无法通过。因此使用 `--check-error-stat max` 会比 `--check-error-stat elemwise` 更严格。

```shell
polygraphy run ../models/ResNet-18.onnx \
    --trt --onnxrt \
    --input-shapes x:[1,3,224,224] \
    --atol 1e-8 --rtol 1e-5 \
    --check-error-stat median
```

<details>
<summary>日志详情</summary>

| mean | std-dev | var | median | min | max | avg-magnitude |
|------|---------|-----|--------|-----|-----|---------------|
| 均值   | 标准差     | 方差  | 中位数    | 最小值 | 最大值 | 平均幅度（绝对值的平均值） |

（1）未通过

```
[I] Accuracy Comparison | trt-runner-N0-03/22/24-14:58:44 vs. onnxrt-runner-N0-03/22/24-14:58:44
[I]     Comparing Output: 'y' (dtype=float32, shape=(1, 1000)) with 'y' (dtype=float32, shape=(1, 1000))
[I]         Tolerance: [abs=1e-05, rel=1e-05] | Checking elemwise error
[I]         trt-runner-N0-03/22/24-14:58:44: y | Stats: mean=1.8082e-06, std-dev=1.4185, var=2.0122, median=-0.099701, min=-4.1133 at (0, 865), max=5.2148 at (0, 111), avg-magnitude=1.1267
[I]             ---- Histogram ----
                Bin Range        |  Num Elems | Visualization
                (-4.11 , -3.18 ) |         10 | #
                (-3.18 , -2.25 ) |         31 | ####
                (-2.25 , -1.31 ) |        131 | ####################
                (-1.31 , -0.382) |        240 | #####################################
                (-0.382, 0.551 ) |        259 | ########################################
                (0.551 , 1.48  ) |        175 | ###########################
                (1.48  , 2.42  ) |        101 | ###############
                (2.42  , 3.35  ) |         38 | #####
                (3.35  , 4.28  ) |         10 | #
                (4.28  , 5.22  ) |          5 | 
[I]         onnxrt-runner-N0-03/22/24-14:58:44: y | Stats: mean=8.1558e-06, std-dev=1.4193, var=2.0144, median=-0.096946, min=-4.1115 at (0, 865), max=5.2154 at (0, 111), avg-magnitude=1.1274
[I]             ---- Histogram ----
                Bin Range        |  Num Elems | Visualization
                (-4.11 , -3.18 ) |         10 | #
                (-3.18 , -2.25 ) |         30 | ####
                (-2.25 , -1.31 ) |        132 | ####################
                (-1.31 , -0.382) |        239 | ####################################
                (-0.382, 0.551 ) |        261 | ########################################
                (0.551 , 1.48  ) |        173 | ##########################
                (1.48  , 2.42  ) |        102 | ###############
                (2.42  , 3.35  ) |         38 | #####
                (3.35  , 4.28  ) |         10 | #
                (4.28  , 5.22  ) |          5 | 
[I]         Error Metrics: y
[I]             Minimum Required Tolerance: elemwise error | [abs=0.012686] OR [rel=7.2541] (requirements may be lower if both abs/rel tolerances are set)
[I]             Absolute Difference | Stats: mean=0.0033219, std-dev=0.0024406, var=5.9566e-06, median=0.0027798, min=1.8477e-05 at (0, 303), max=0.012686 at (0, 983), avg-magnitude=0.0033219
[I]                 ---- Histogram ----
                    Bin Range           |  Num Elems | Visualization
                    (1.85e-05, 0.00129) |        231 | ########################################
                    (0.00129 , 0.00255) |        231 | ########################################
                    (0.00255 , 0.00382) |        170 | #############################
                    (0.00382 , 0.00509) |        150 | #########################
                    (0.00509 , 0.00635) |         91 | ###############
                    (0.00635 , 0.00762) |         56 | #########
                    (0.00762 , 0.00889) |         39 | ######
                    (0.00889 , 0.0102 ) |         23 | ###
                    (0.0102  , 0.0114 ) |          8 | #
                    (0.0114  , 0.0127 ) |          1 | 
[I]             Relative Difference | Stats: mean=0.023228, std-dev=0.25767, var=0.066395, median=0.0028818, min=7.1108e-06 at (0, 255), max=7.2541 at (0, 217), avg-magnitude=0.023228
[I]                 ---- Histogram ----
                    Bin Range         |  Num Elems | Visualization
                    (7.11e-06, 0.725) |        996 | ########################################
                    (0.725   , 1.45 ) |          1 | 
                    (1.45    , 2.18 ) |          1 | 
                    (2.18    , 2.9  ) |          0 | 
                    (2.9     , 3.63 ) |          1 | 
                    (3.63    , 4.35 ) |          0 | 
                    (4.35    , 5.08 ) |          0 | 
                    (5.08    , 5.8  ) |          0 | 
                    (5.8     , 6.53 ) |          0 | 
                    (6.53    , 7.25 ) |          1 | 
[E]         FAILED | Output: 'y' | Difference exceeds tolerance (rel=1e-05, abs=1e-05)
[E]     FAILED | Mismatched outputs: ['y']
[E] Accuracy Summary | trt-runner-N0-03/22/24-14:58:44 vs. onnxrt-runner-N0-03/22/24-14:58:44 | Passed: 0/1 iterations | Pass Rate: 0.0%
```

（2）通过

启用 `--verbose` 选项，可以得到类似未通过时的详细数据分布信息。
```
[I] Accuracy Comparison | trt-runner-N0-03/22/24-17:07:24 vs. onnxrt-runner-N0-03/22/24-17:07:24
[I]     Comparing Output: 'y' (dtype=float32, shape=(1, 1000)) with 'y' (dtype=float32, shape=(1, 1000))
[I]         Tolerance: [abs=1e-05, rel=1e-05] | Checking elemwise error
[I]         trt-runner-N0-03/22/24-17:07:24: y | Stats: mean=8.1482e-06, std-dev=1.4193, var=2.0144, median=-0.096945, min=-4.1115 at (0, 865), max=5.2154 at (0, 111), avg-magnitude=1.1274
[I]         onnxrt-runner-N0-03/22/24-17:07:24: y | Stats: mean=8.1558e-06, std-dev=1.4193, var=2.0144, median=-0.096946, min=-4.1115 at (0, 865), max=5.2154 at (0, 111), avg-magnitude=1.1274
[I]         Error Metrics: y
[I]             Minimum Required Tolerance: elemwise error | [abs=2.3842e-06] OR [rel=0.00077459] (requirements may be lower if both abs/rel tolerances are set)
[I]             Absolute Difference | Stats: mean=5.1292e-07, std-dev=4.0099e-07, var=1.6079e-13, median=4.3193e-07, min=0 at (0, 3), max=2.3842e-06 at (0, 451), avg-magnitude=5.1292e-07
[I]             Relative Difference | Stats: mean=2.4714e-06, std-dev=2.6136e-05, var=6.831e-10, median=4.4903e-07, min=0 at (0, 3), max=0.00077459 at (0, 217), avg-magnitude=2.4714e-06
[I]         PASSED | Output: 'y' | Difference is within tolerance (rel=1e-05, abs=1e-05)
[I]     PASSED | All outputs matched | Outputs: ['y']
[I] Accuracy Summary | trt-runner-N0-03/22/24-17:07:24 vs. onnxrt-runner-N0-03/22/24-17:07:24 | Passed: 1/1 iterations | Pass Rate: 100.0%
```

</details>

在实际应用中，输出结果不匹配可能是在某些层出现了异常。
`--trt-outputs` 和 `--onnx-outputs` 选项可以接收输出名称作为参数，对比特定层的结果。
`mark all` 作为参数可以对比所有张量，配合 `--fail-fast` 选项可以在出现第一个不匹配的位置退出，便于找到问题的根源。

```shell
polygraphy run ../models/ResNet-18.onnx \
    --trt --onnxrt \
    --input-shapes x:[1,3,224,224] \
    --atol 1e-8 --rtol 1e-8 \
    --trt-outputs /layer4/layer4.1/conv2/Conv_output_0 /avgpool/GlobalAveragePool_output_0 \
    --onnx-outputs /layer4/layer4.1/conv2/Conv_output_0 /avgpool/GlobalAveragePool_output_0 
```

```shell
polygraphy run ../models/ResNet-18.onnx \
    --trt --onnxrt \
    --input-shapes x:[1,3,224,224] \
    --atol 1e-8 --rtol 1e-8 \
    --trt-outputs mark all \
    --onnx-outputs mark all \
    --fail-fast
```

注意：`--trt-outputs mark all` 有时会因为不同的时序、层融合选择、格式约束
（differences in timing, layer fusion choices, and format constraints）
影响生成的引擎，从而规避了输出不匹配的情况。此时可能要使用更复杂的方法来切分模型并生成复现错误的测试用例。

# 2. 保存输入输出
保存输入和输出值，第二次运行时加载输入作为输入，加载输出作为对比数据。
```shell
polygraphy run ../models/ResNet-18.onnx \
    --onnxrt \
    --save-inputs inputs.json \
    --save-outputs outputs.json
```

```shell
polygraphy run ../models/ResNet-18.onnx \
    --trt \
    --load-inputs inputs.json \
    --load-outputs outputs.json
```

只要输入输出相匹配，还可以用这种方式直接对比 TensorRT 引擎和 ONNX。
首先把 ONNX 转为 TensorRT 引擎并保存，然后使用之前 ONNX-Runtime 保存的输入输出进行对比。

```shell
polygraphy convert ../models/ResNet-18.onnx \
    -o ResNet-18.engine
```

```shell
polygraphy run --trt ResNet-18.engine \
    --model-type=engine \
    --load-inputs inputs.json \
    --load-outputs outputs.json
```

# 3. 生成 Python 脚本
非常方便的一个功能，当 CIL 命令不足以满足需求时，可以先生成 Python 脚本，满足一些基础需求，
然后在此基础上修改脚本，能更便捷的满足更复杂的功能；
或者在使用 Polygraphy Python API 时有些功能不知道怎么实现，
但用 CIL 能实现时，可以直接生成脚本，学习 Python API 的使用方法。 

```shell
polygraphy run ../models/ResNet-18.onnx \
    --trt --onnxrt \
    --input-shapes x:[1,3,224,224] \
    --atol 1e-8 --rtol 1e-8 \
    --trt-outputs  mark all \
    --onnx-outputs  mark all \
    --gen-script=compare_trt_onnxrt.py
```

# 4. 检查异常值 NaN 和 Inf 

官方给出的示例模型较为简单，这里使用 [ResNet_div_zero.py](..%2Fgenerate_onnx%2FResNet_div_zero.py) 
在 ResNet-18 的最后一个全连接层后面新增一个 Div 节点，进行除0操作。
`-vv` 用于显示详细信息。

```shell
python ResNet_div_zero.py
```
```shell
polygraphy run ../models/ResNet_div_zero.onnx \
    --onnxrt --validate \
    -vv
```
```shell
polygraphy run ../models/ResNet_div_zero.onnx \
    --trt --validate \
    -vv
```