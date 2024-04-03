
from polygraphy import mod
from polygraphy import func
from polygraphy.backend.trt import CreateConfig as CreateTrtConfig
trt = mod.lazy_import('tensorrt')

create_trt_config = CreateTrtConfig()


@func.extend(create_trt_config)
def load_config(builder, network, config):
    config.set_flag(trt.BuilderFlag.FP16)
