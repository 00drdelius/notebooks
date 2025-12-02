"""
implement RoPE, rotary position embeddings.
Refer more to the following url:
1. original paper: https://arxiv.org/pdf/2104.09864
2. improved implementation: https://nn.labml.ai/transformers/rope/index.html
3. different implementation styles brief: https://anyinlover.github.io/blog/deep_learning/deepseek_rope
"""

from typing import *

import torch
import torch.nn as nn
from transformers.models.qwen2 import Qwen2Config

my_llm_config = Qwen2Config(
    vocab_size=151936,
    hidden_size=512,
    intermediate_size=4096*3,
    num_attention_heads=16,
    num_key_value_heads=4,
    hidden_act="silu",
    max_position_embeddings=32768,
    use_cache=False,
    tie_word_embeddings=False,
    use_sliding_window=False,

    attn_bias = False,
    rope_type="default",
)
CallableModule: TypeAlias = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

class BaseRoPE(nn.Module):
    # 2. Type hint the buffer to fix linting/MyPy errors regarding 'register_buffer'
    inv_freq: torch.Tensor

    def __init__(self, llm_config:Qwen2Config):
        super().__init__()
        self.head_dim = getattr(llm_config, "head_dim", llm_config.hidden_size // llm_config.num_attention_heads)
        self.max_position_embeddings = llm_config.max_position_embeddings
        self.rope_theta = llm_config.rope_theta
        self._get_position_embeddings()

    def _get_position_embeddings(self):
        """
        calculate inverse frequencies:
        $$
        \\frac{1} \\
        {rope_theta^{\\frac{2(i-1)}{d}}},
        i \\in [1, 2, ..., head_dim//2]
        $$
        """
        inv_freq = torch.pow(self.rope_theta, torch.arange(0, self.head_dim//2, dtype=torch.int32)*2/self.head_dim)
        inv_freq = torch.pow(inv_freq.float(), -1)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """
        construct:
        1. cos[
        [pos_i\*theta_1, pos_i\*theta_2, ..., pos_i\*theta_{d/2}, pos_i\*theta_1, pos_i\*theta_2, ..., pos_i\*theta_{d/2}]
        , ...],

        2. sin[
        [pos_i\*theta_1, pos_i\*theta_2, ..., pos_i\*theta_{d/2}, pos_i\*theta_1, pos_i\*theta_2, ..., pos_i\*theta_{d/2}]
        , ...],

        pos_i \\in [0, 1, 2, ..., seq_len]
        """
        bsz, seq_len, _ = hidden_states.shape
        
        # expand inverse frequencies to [bsz, 1, head_dim]
        expanded_inv_freq = self.inv_freq[None, None, :].expand(*[bsz, 1, -1])
        # expand position ids to [bsz, seq_len, 1]
        expanded_position_ids = torch.arange(
            seq_len)[None, :, None].expand(*[bsz, -1, 1]).float()
        print("[size of expanded inv_freq] ",expanded_inv_freq.shape)
        print("[size of expanded_position_ids] ",expanded_position_ids.shape)

        with torch.autocast(device_type=hidden_states.device.type, dtype=torch.float32):
            #NOTE you don't need to manually convert tensor dtype to float32 and device to the same here,
            # torch.autocast does these conversions automatically.
            half_idx_theta = expanded_position_ids @ expanded_inv_freq
            idx_theta = torch.cat([half_idx_theta, half_idx_theta], dim=-1)
        
        cos = idx_theta.cos().to(dtype=hidden_states.dtype)
        sin = idx_theta.sin().to(dtype=hidden_states.dtype)
        return cos, sin

    @classmethod
    def apply_position_embeddings(cls, head_states:torch.Tensor, position_embeddings: tuple[torch.Tensor,torch.Tensor]):
        """
        calculate:
        [
        [cos pos_i \* theta, -sin pos_i \* theta],
        [sin pos_i \* theta, cos pos_i \* theta]
        ]

        @

        [
        x^{(pos_i)},
        -x^{(pos_i + head_dim//2)}
        ]
        """
        bsz, attn_heads, seq_len, head_dim = head_states.shape
        cos, sin = position_embeddings
        expanded_cos = cos[:, None, :, :].expand(*[-1, attn_heads, -1, -1])        
        expaned_sin = sin[:, None, :, :].expand(*[-1, attn_heads, -1, -1])

        apply_cos = head_states * expanded_cos
        
        def rotate_half(x:torch.Tensor):
            return torch.cat([-x[..., head_dim//2:], x[..., :head_dim//2]], dim=-1)

        apply_sin = rotate_half(head_states) * expaned_sin

        return apply_cos+apply_sin


from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb

# test precision
def test():
    torch.set_default_device("cuda:0")
    my_rotary_embd = BaseRoPE(my_llm_config)
    qwen2_rotary_embd = Qwen2RotaryEmbedding(my_llm_config)

    # test inv freq
    print("[qwen2 inv freq] ", qwen2_rotary_embd.inv_freq)
    print("[my inv freq] ", my_rotary_embd.inv_freq)
    inv_freq_equal = torch.allclose(qwen2_rotary_embd.inv_freq, my_rotary_embd.inv_freq)
    print("[inv_freq equal] ",inv_freq_equal)

    # test cos and sin
    bsz, seq_len, hidden_size = 5, 1024, my_llm_config.hidden_size
    hidden_states = torch.randn(size=[bsz, seq_len, hidden_size], dtype=torch.bfloat16)
    my_cos, my_sin = my_rotary_embd(hidden_states)

    position_ids=torch.arange(seq_len)[None, :].expand(*[bsz,-1])
    qwen2_cos, qwen2_sin = qwen2_rotary_embd(hidden_states,position_ids)

    print("[cos equal] ",torch.allclose(my_cos, qwen2_cos))
    print("[sin equal] ", torch.allclose(my_sin, qwen2_sin))

    # test apply position embeddings
    head_dim = my_rotary_embd.head_dim
    query_states = torch.randn(bsz, my_llm_config.num_attention_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key_states = torch.randn(bsz, my_llm_config.num_key_value_heads, seq_len, head_dim, dtype=torch.bfloat16)
    print("[query_states shape] ",query_states.shape)
    print("[key_states shape] ",key_states.shape)

    qwen2_query_states, qwen2_key_states = apply_rotary_pos_emb(query_states, key_states, qwen2_cos, qwen2_sin)

    my_query_states = my_rotary_embd.apply_position_embeddings(query_states, (my_cos, my_sin))
    my_key_states = my_rotary_embd.apply_position_embeddings(key_states, (my_cos,my_sin))

    print("[query states equal] ",torch.allclose(qwen2_query_states, my_query_states))
    print("[key states equal] ",torch.allclose(qwen2_key_states, my_key_states))

test()