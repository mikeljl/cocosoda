from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer)
model_name = "DeepSoftwareAnalytics/CoCoSoDa"

cache_dir = "./hf_cache"

config = RobertaConfig.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = RobertaModel.from_pretrained(model_name, cache_dir=cache_dir) 