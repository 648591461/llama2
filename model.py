import math
from typing import List, Optional, Tuple, Union

from vocab import vocab
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, LLAMA_START_DOCSTRING

logger = logging.get_logger('llama')

config = LlamaConfig(
    vocab_size=len(vocab),  # 语料库大小
    hidden_size=512,  # 隐藏层大小
    intermediate_size=2752,  # ？？？
    num_hidden_layers=8,  # 层数
    num_attention_heads=16,  # 注意力头 数
    hidden_act='silu',  # 不清楚
    max_position_embeddings=128,  # 位置编码
    initializer_range=0.02,  # 初始化范围
    rms_norm_eps=1e-06,  # ??? 涉及到优化
    use_cache=True,  # 使用缓存
    pad_token_id=0,  # 填充token
    bos_token_id=1,  # ???
    eos_token_id=2,  # 终止token
    tie_word_embeddings=False  # ？？？
)

# 旋转位置编码 RoPE
