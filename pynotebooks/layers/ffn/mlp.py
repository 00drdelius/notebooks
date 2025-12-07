import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe import Qwen3MoeConfig

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


class BaseMLP(nn.Module):
    def __init__(self, llm_config:Qwen3MoeConfig):
        super().__init__()
        self.gate_proj = nn.Linear(llm_config.hidden_size, llm_config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(llm_config.hidden_size, llm_config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(llm_config.intermediate_size, llm_config.hidden_size, bias=False)
        match llm_config.hidden_act:
            case _:
                self.hidden_act = F.silu # set silu as default
        
    def forward(self, hidden_states:torch.Tensor):
        hidden_states = self.down_proj(self.hidden_act(self.gate_proj(hidden_states) * self.gate_proj(hidden_states)))
        return hidden_states

# test precision
"""
don't think there's any necessary to match MLP precision
"""
