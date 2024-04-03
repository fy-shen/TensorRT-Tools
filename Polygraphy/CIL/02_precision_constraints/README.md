# 指定层的精度

首先创建一个模型用于此示例，模型执行以下操作：
1. 右乘副对角线为1的矩阵
2. 加上 FP16 精度最大值
3. 减去 FP16 精度最大值
4. 右乘副对角线为1的矩阵

```shell
python create_network.py
```

## （1）使用 `--layer-precisions` 指定各层精度

强制 `Add` 层使用 FP16 精度，`Sub` 层使用 FP32 精度，
防止层融合，顺利引发数值溢出。

```shell
polygraphy run needs_constraints.onnx \
    --trt --fp16 --onnxrt --val-range x:[1,2] \
    --layer-precisions Add:float16 Sub:float32 \
    --precision-constraints prefer \
    --trt-outputs mark all \
    --onnx-outputs mark all \
    --check-error-stat median > log.log
```
观察日志信息，从 `Add` 层引入了绝对误差 1\~2，
`Sub` 层输出在 FP16 下为 0\~0.2，在 FP32 下为 1\~2。

<details>
<summary>add_out_1</summary>

```
[I]     Comparing Output: 'add_out_1' (dtype=float32, shape=(1, 1, 256, 256)) with 'add_out_1' (dtype=float32, shape=(1, 1, 256, 256))
[I]         Tolerance: [abs=1e-05, rel=1e-05] | Checking median error
[I]         trt-runner-N0-04/01/24-15:29:00: add_out_1 | Stats: mean=65504, std-dev=0, var=0, median=65504, min=65504 at (0, 0, 0, 0), max=65504 at (0, 0, 0, 0), avg-magnitude=65504
[I]             ---- Histogram ----
                Bin Range            |  Num Elems | Visualization
                (6.55e+04, 6.55e+04) |      65536 | ########################################
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
[I]         onnxrt-runner-N0-04/01/24-15:29:00: add_out_1 | Stats: mean=65506, std-dev=0.28889, var=0.083457, median=65506, min=65505 at (0, 0, 0, 5), max=65506 at (0, 0, 5, 165), avg-magnitude=65506
[I]             ---- Histogram ----
                Bin Range            |  Num Elems | Visualization
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |          0 | 
                (6.55e+04, 6.55e+04) |      12931 | ######################################
                (6.55e+04, 6.55e+04) |      13129 | #######################################
                (6.55e+04, 6.55e+04) |      13276 | ########################################
                (6.55e+04, 6.55e+04) |      13061 | #######################################
                (6.55e+04, 6.55e+04) |      13139 | #######################################
[I]         Error Metrics: add_out_1
[I]             Minimum Required Tolerance: median error | [abs=1.5] OR [rel=2.2899e-05]
[I]             Absolute Difference | Stats: mean=1.4995, std-dev=0.28889, var=0.083457, median=1.5, min=1 at (0, 0, 0, 5), max=2 at (0, 0, 5, 165), avg-magnitude=1.4995
[I]                 ---- Histogram ----
                    Bin Range  |  Num Elems | Visualization
                    (1  , 1.1) |       6588 | #######################################
                    (1.1, 1.2) |       6616 | #######################################
                    (1.2, 1.3) |       6499 | ######################################
                    (1.3, 1.4) |       6593 | #######################################
                    (1.4, 1.5) |       6291 | #####################################
                    (1.5, 1.6) |       6749 | ########################################
                    (1.6, 1.7) |       6678 | #######################################
                    (1.7, 1.8) |       6383 | #####################################
                    (1.8, 1.9) |       6619 | #######################################
                    (1.9, 2  ) |       6520 | ######################################
[I]             Relative Difference | Stats: mean=2.2892e-05, std-dev=4.4101e-06, var=1.9449e-11, median=2.2899e-05, min=1.5266e-05 at (0, 0, 0, 5), max=3.0532e-05 at (0, 0, 5, 165), avg-magnitude=2.2892e-05
[I]                 ---- Histogram ----
                    Bin Range            |  Num Elems | Visualization
                    (1.53e-05, 1.68e-05) |       6588 | #######################################
                    (1.68e-05, 1.83e-05) |       6616 | #######################################
                    (1.83e-05, 1.98e-05) |       6499 | ######################################
                    (1.98e-05, 2.14e-05) |       6593 | #######################################
                    (2.14e-05, 2.29e-05) |       6291 | #####################################
                    (2.29e-05, 2.44e-05) |       6749 | ########################################
                    (2.44e-05, 2.6e-05 ) |       6678 | #######################################
                    (2.6e-05 , 2.75e-05) |       6383 | #####################################
                    (2.75e-05, 2.9e-05 ) |       6619 | #######################################
                    (2.9e-05 , 3.05e-05) |       6520 | ######################################
[E]         FAILED | Output: 'add_out_1' | Difference exceeds tolerance (rel=1e-05, abs=1e-05)
```
</details>

<details>
<summary>sub_out_2</summary>

```
[I]     Comparing Output: 'sub_out_2' (dtype=float32, shape=(1, 1, 256, 256)) with 'sub_out_2' (dtype=float32, shape=(1, 1, 256, 256))
[I]         Tolerance: [abs=1e-05, rel=1e-05] | Checking median error
[I]         trt-runner-N0-04/01/24-15:29:00: sub_out_2 | Stats: mean=0, std-dev=0, var=0, median=0, min=0 at (0, 0, 0, 0), max=0 at (0, 0, 0, 0), avg-magnitude=0
[I]             ---- Histogram ----
                Bin Range  |  Num Elems | Visualization
                (0  , 0.2) |      65536 | ########################################
                (0.2, 0.4) |          0 | 
                (0.4, 0.6) |          0 | 
                (0.6, 0.8) |          0 | 
                (0.8, 1  ) |          0 | 
                (1  , 1.2) |          0 | 
                (1.2, 1.4) |          0 | 
                (1.4, 1.6) |          0 | 
                (1.6, 1.8) |          0 | 
                (1.8, 2  ) |          0 | 
[I]         onnxrt-runner-N0-04/01/24-15:29:00: sub_out_2 | Stats: mean=1.4995, std-dev=0.28889, var=0.083457, median=1.5, min=1 at (0, 0, 0, 5), max=2 at (0, 0, 5, 165), avg-magnitude=1.4995
[I]             ---- Histogram ----
                Bin Range  |  Num Elems | Visualization
                (0  , 0.2) |          0 | 
                (0.2, 0.4) |          0 | 
                (0.4, 0.6) |          0 | 
                (0.6, 0.8) |          0 | 
                (0.8, 1  ) |          0 | 
                (1  , 1.2) |      13204 | ########################################
                (1.2, 1.4) |      13092 | #######################################
                (1.4, 1.6) |      13040 | #######################################
                (1.6, 1.8) |      13061 | #######################################
                (1.8, 2  ) |      13139 | #######################################
[I]         Error Metrics: sub_out_2
[I]             Minimum Required Tolerance: median error | [abs=1.5] OR [rel=1]
[I]             Absolute Difference | Stats: mean=1.4995, std-dev=0.28889, var=0.083457, median=1.5, min=1 at (0, 0, 0, 5), max=2 at (0, 0, 5, 165), avg-magnitude=1.4995
[I]                 ---- Histogram ----
                    Bin Range  |  Num Elems | Visualization
                    (1  , 1.1) |       6588 | #######################################
                    (1.1, 1.2) |       6616 | #######################################
                    (1.2, 1.3) |       6499 | ######################################
                    (1.3, 1.4) |       6593 | #######################################
                    (1.4, 1.5) |       6291 | #####################################
                    (1.5, 1.6) |       6749 | ########################################
                    (1.6, 1.7) |       6678 | #######################################
                    (1.7, 1.8) |       6383 | #####################################
                    (1.8, 1.9) |       6619 | #######################################
                    (1.9, 2  ) |       6520 | ######################################
[I]             Relative Difference | Stats: mean=1, std-dev=0, var=0, median=1, min=1 at (0, 0, 0, 0), max=1 at (0, 0, 0, 0), avg-magnitude=1
[I]                 ---- Histogram ----
                    Bin Range  |  Num Elems | Visualization
                    (0.5, 0.6) |          0 | 
                    (0.6, 0.7) |          0 | 
                    (0.7, 0.8) |          0 | 
                    (0.8, 0.9) |          0 | 
                    (0.9, 1  ) |          0 | 
                    (1  , 1.1) |      65536 | ########################################
                    (1.1, 1.2) |          0 | 
                    (1.2, 1.3) |          0 | 
                    (1.3, 1.4) |          0 | 
                    (1.4, 1.5) |          0 | 
[E]         FAILED | Output: 'sub_out_2' | Difference exceeds tolerance (rel=1e-05, abs=1e-05)
```
</details>


## （2）使用后处理脚本

```shell
polygraphy run needs_constraints.onnx \
    --onnxrt --trt --fp16 --precision-constraints obey \
    --val-range x:[1,2] --check-error-stat median \
    --trt-network-postprocess-script ./postprocess.py
```

## （3）使用网络构建器脚本
```shell
polygraphy run needs_constraints.onnx \
    --onnxrt --val-range x:[1,2] \
    --save-inputs inputs.json \
    --save-outputs outputs.json
```

```shell
polygraphy run loader.py --precision-constraints obey \
    --trt --fp16 \
    --load-inputs inputs.json \
    --load-outputs outputs.json \
    --check-error-stat median
```

# 测试

```bash
bash test.sh
```
总结一些现象（复现结果可能会与 TensorRT 版本和硬件相关）

1. 第二项测试，即仅在 CIL 中启用 `--fp16` 时，得到的引擎精度依然是 FP32；
2. 三种方式下强制 `Add` 和 `Sub` 层为 FP16，都会使 FP16 生效，但层会产生融合，导致不会出现数值溢出的问题；
3. 观察 `tactics` 相关日志信息，TensorRT 所使用的优化策略有所不同。
在引擎构建过程中，TensorRT 会运行多种 kernel 来选择最佳的，但每次运行可能略有不同，
导致优化策略并不固定。可以使用 `--save-tactics` 和 `--load-tactics` 来保存和加载优化策略；