"""
implement RoPE, rotary position embeddings.
Refer more to the following url:
1. original paper: https://arxiv.org/pdf/2104.09864
2. improved implementation: https://nn.labml.ai/transformers/rope/index.html
3. different implementation styles brief: https://anyinlover.github.io/blog/deep_learning/deepseek_rope
"""

import torch
import torch.nn as nn

from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

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

    attn_bias = False
)


class BaseRoPE(nn.Module):
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
        inv_freq = torch.pow(self.rope_theta, torch.arange(0, self.head_dim//2, dtype=torch.int32)*2//self.head_dim)
        self.inv_freq = torch.pow(inv_freq.float(torch.float32), -1)
        self.register_buffer("inv_freq", self.inv_freq)

    def forward(self, hidden_states:torch.Tensor):
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
            seq_len)[None, :, None].expand(*[bsz, -1, 1])

        with torch.autocast(device_type=hidden_states.device, dtype=torch.float32):
            #NOTE you don't need to manually convert tensor dtype to float32 and device to the same here,
            # torch.autocast does these conversions automatically.
            half_idx_theta = expanded_position_ids @ expanded_inv_freq
            idx_theta = torch.cat([half_idx_theta, half_idx_theta], dim=-1)
        
        cos = idx_theta.cos()
        sin = idx_theta.sin()
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
        bsz, attn_heads, seq_len, head_dim = head_states
        cos, sin = position_embeddings
        expanded_cos = cos[:, None, :, :].expand(*[-1, attn_heads, -1, -1])        
        expaned_sin = sin[:, None, :, :].expand(*[-1, head_dim, -1, -1])

        apply_cos = head_states * expanded_cos
        
        def rotate_half(x:torch.Tensor):
            return torch.cat([-x[:, head_dim//2:], x[:, :head_dim//2]], dim=-1)

        apply_sin = rotate_half(head_states) * expaned_sin

        return apply_cos+apply_sin