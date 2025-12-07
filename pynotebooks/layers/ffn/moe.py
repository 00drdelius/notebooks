import torch
import torch.nn

from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

# general
vocab_size = 151936
hidden_size = 512

# rope
max_position_embeddings = 32768
rope_theta=10000.0

# attention
num_attention_heads = 32
num_key_value_heads = 4
use_sliding_window = False
attn_bias = False

# ffn
hidden_act = "silu"
num_experts_per_token = 8
num_experts=128

# lm head
tie_word_embeddings = False

my_llm_moe_config = Qwen3MoeConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,

    rope_theta=rope_theta,
    max_position_embeddings=max_position_embeddings,

    use_sliding_window=use_sliding_window,
    num_attention_heads=num_attention_heads,
    num_key_value_heads=num_key_value_heads,
    attn_bias = attn_bias,

    hidden_act=hidden_act,
    intermediate_size=hidden_size*3,
    moe_intermediate_size=hidden_size*3//num_experts_per_token,
    num_experts_per_tok=num_experts_per_token,
    num_experts=128,

    tie_word_embeddings=tie_word_embeddings,

    use_cache=False,
)



