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
num_experts=32

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
    num_experts=num_experts,

    tie_word_embeddings=tie_word_embeddings,

    use_cache=False,
)

def test():
    torch.set_default_device("cuda:0")
    qwen3moe_block = Qwen3MoeSparseMoeBlock(config=my_llm_moe_config).eval()
    print(qwen3moe_block)
    bsz, seq_len, hidden_size = 4, 1024, my_llm_moe_config.hidden_size
    hidden_states = torch.randn(size=(bsz, seq_len, hidden_size), dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda:0",dtype=torch.bfloat16):
        output = qwen3moe_block(hidden_states)

    print(output)

test()