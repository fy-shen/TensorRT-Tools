#/bin/bash

set -e
set -x

clear
mkdir -p log engine tactics
rm -rf log/*.log engine/*.engine tactics/*.json

# 1. FP32
polygraphy convert needs_constraints.onnx \
    --save-tactics tactics/fp32.json \
    -o engine/fp32.engine \
    >log/convert_fp32.log

polygraphy inspect model engine/fp32.engine --show layers >log/model_fp32.log
polygraphy inspect tactics tactics/fp32.json >log/tactics_fp32.log

# 2. FP16
polygraphy convert needs_constraints.onnx \
    --fp16 \
    --save-tactics tactics/fp16.json \
    -o engine/fp16.engine \
    >log/convert_fp16.log

polygraphy inspect model engine/fp16.engine --show layers >log/model_fp16.log
polygraphy inspect tactics tactics/fp16.json >log/tactics_fp16.log

# 3. `--layer-precisions` 强制 FP16+FP32
polygraphy convert needs_constraints.onnx \
    --fp16 \
    --layer-precisions Add:float16 Sub:float32 \
    --precision-constraints prefer \
    --save-tactics tactics/fp16_fp32.json \
    -o engine/fp16_fp32.engine \
    >log/convert_fp16_fp32.log

polygraphy inspect model engine/fp16_fp32.engine --show layers >log/model_fp16_fp32.log
polygraphy inspect tactics tactics/fp16_fp32.json >log/tactics_fp16_fp32.log

# 4. `--layer-precisions` 强制 FP16+FP16
polygraphy convert needs_constraints.onnx \
    --fp16 \
    --layer-precisions Add:float16 Sub:float16 \
    --precision-constraints prefer \
    --save-tactics tactics/fp16_fp16.json \
    -o engine/fp16_fp16.engine \
    >log/convert_fp16_fp16.log

polygraphy inspect model engine/fp16_fp16.engine --show layers >log/model_fp16_fp16.log
polygraphy inspect tactics tactics/fp16_fp16.json >log/tactics_fp16_fp16.log

# 5. `--trt-npps` 强制 FP16+FP32
polygraphy convert needs_constraints.onnx \
    --fp16 \
    --trt-npps postprocess.py \
    --precision-constraints prefer \
    --save-tactics tactics/fp16_fp32_npps.json \
    -o engine/fp16_fp32_npps.engine \
    >log/convert_fp16_fp32_npps.log

polygraphy inspect model engine/fp16_fp32_npps.engine --show layers >log/model_fp16_fp32_npps.log
polygraphy inspect tactics tactics/fp16_fp32_npps.json >log/tactics_fp16_fp32_npps.log

# 6. `--trt-npps` 强制 FP16+FP16
polygraphy convert needs_constraints.onnx \
    --fp16 \
    --trt-npps postprocess.py:postprocess_fp16 \
    --precision-constraints prefer \
    --save-tactics tactics/fp16_fp16_npps.json \
    -o engine/fp16_fp16_npps.engine \
    >log/convert_fp16_fp16_npps.log

polygraphy inspect model engine/fp16_fp16_npps.engine --show layers >log/model_fp16_fp16_npps.log
polygraphy inspect tactics tactics/fp16_fp16_npps.json >log/tactics_fp16_fp16_npps.log

# 7. trt-network-script 强制 FP16+FP32
polygraphy convert loader.py \
    --fp16 \
    --precision-constraints prefer \
    --save-tactics tactics/fp16_fp32_loader.json \
    -o engine/fp16_fp32_loader.engine \
    >log/convert_fp16_fp32_loader.log

polygraphy inspect model engine/fp16_fp32_loader.engine --show layers >log/model_fp16_fp32_loader.log
polygraphy inspect tactics tactics/fp16_fp32_loader.json >log/tactics_fp16_fp32_loader.log

# 8. trt-network-script 强制 FP16+FP16
polygraphy convert loader.py:load_network_fp16 \
    --fp16 \
    --precision-constraints prefer \
    --save-tactics tactics/fp16_fp16_loader.json \
    -o engine/fp16_fp16_loader.engine \
    >log/convert_fp16_fp16_loader.log

polygraphy inspect model engine/fp16_fp16_loader.engine --show layers >log/model_fp16_fp16_loader.log
polygraphy inspect tactics tactics/fp16_fp16_loader.json >log/tactics_fp16_fp16_loader.log
