"""
implement general GQA, tested result with Qwen2 official GQA implementation
"""
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

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


TypedLinear: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
TypedModule: TypeAlias = Callable[[torch.Tensor], Sequence[torch.Tensor]]

class BaseAttention(nn.Module):
    "Base GQA without RoPE"
    def __init__(self, config:Qwen2Config, ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        print("base head_dim: ",self.head_dim)

        #NOTE here must be `head_dim^{-0.5}` cuz you get different results
        # between `/ head_dim^{0.5}` and `* head_dim^{-0.5}` as division and multiplication in computer are not equal to real math.
        self.scaling = self.head_dim**-0.5
        allow_bias = getattr(config, "attn_bias", False)

        self.q_proj:TypedLinear = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=allow_bias, dtype=torch.float16)
        self.k_proj:TypedLinear = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=allow_bias, dtype=torch.float16)
        self.v_proj:TypedLinear = nn.Linear(self.hidden_size, self.kv_heads * self.head_dim, bias=allow_bias, dtype=torch.float16)
        self.o_proj:TypedLinear = nn.Linear(self.num_heads*self.head_dim, self.hidden_size, bias=allow_bias, dtype=torch.float16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor=None
    )->tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        attn_outputs = O(softmax(Q*K.T) * V).

        Args:
            attention_mask(torch.Tensor): It's used in training, \
                or batch sequences ready to be inferred with left or right padding 
        """
        input_shape = hidden_states.shape[:-1] # [batch_size, seq_len, hidden_dim]
        attn_hidden_shape = (*input_shape, -1, self.head_dim) # [batch_size, seq_len, (num_heads or kv_heads), head_dim]

        # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, (num_heads or kv_heads), head_dim]
        # tranpose -> [batch_size, (num_heads or kv_heads), seq_len, head_dim]
        query_states = self.q_proj(hidden_states).view(attn_hidden_shape).transpose(1,2)
        key_states = self.k_proj(hidden_states).view(attn_hidden_shape).transpose(1,2)
        value_states = self.v_proj(hidden_states).view(attn_hidden_shape).transpose(1,2)

        batch_size, _, seq_len, _ = key_states.shape
        # GQA method 1: expand group of [kv_heads, seq_len, head_dim] to num_kv_groups
        # key_states = key_states.unsqueeze(1).expand(
        #     batch_size, self.num_kv_groups, self.kv_heads, seq_len, self.head_dim).reshape(
        #         batch_size, self.num_kv_groups*self.kv_heads, seq_len, self.head_dim)
        # value_states = value_states.unsqueeze(1).expand(
        #     batch_size, self.num_kv_groups, self.kv_heads, seq_len, self.head_dim).reshape(
        #         batch_size, self.num_kv_groups*self.kv_heads, seq_len, self.head_dim)

        # GQA method 2: expand group of [seq_len, head_dim] to num_kv_groups
        #NOTE Qwen2 use this expansion 
        key_states = key_states.unsqueeze(2).expand(
            batch_size, self.kv_heads, self.num_kv_groups, seq_len, self.head_dim).reshape(
                batch_size, self.num_kv_groups*self.kv_heads, seq_len, self.head_dim)
        value_states = value_states.unsqueeze(2).expand(
            batch_size, self.kv_heads, self.num_kv_groups, seq_len, self.head_dim).reshape(
                batch_size, self.num_kv_groups*self.kv_heads, seq_len, self.head_dim)

        # Q * K.T -> [batch_size, (num_heads or kv_heads), seq_len, seq_len]
        # softmax(Q*K.T) * V -> [batch_size, (num_heads or kv_heads), seq_len, head_dim]
        attn_weights_before_softmax = (query_states @ key_states.transpose(-1,-2)) * self.scaling
        # attn_weights_before_softmax = torch.matmul(query_states, key_states.transpose(2,3)) / self.scaling

        #NOTE you can optimize softmax called online normalizer, refer more info to https://arxiv.org/pdf/1805.02867
        attn_weights = F.softmax(attn_weights_before_softmax, dim=-1, dtype=torch.float32).to(query_states.dtype)
        score_tensor = attn_weights @ value_states

        # [batch_size, (num_heads or kv_heads), seq_len, head_dim] -> [batch_size, seq_len, (num_heads or kv_heads), head_dim]
        score_tensor = score_tensor.transpose(1, 2).contiguous()
        # [batch_size, seq_len, (num_heads or kv_heads), head_dim] -> [batch_size, seq_len, (num_heads or kv_heads)*head_dim]
        score_tensor =score_tensor.reshape(batch_size, seq_len, -1)

        attn_output = self.o_proj(score_tensor)
        
        return dict(
            query_states=query_states,key_states=key_states,value_states=value_states,
            attn_output=attn_output, attn_weights=attn_weights_before_softmax
        )


# qwen2 attention without RoPE and mask
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights_before_softmax = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # if attention_mask is not None:
    #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #     attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(
        attn_weights_before_softmax, dim=-1,
        dtype=torch.float32
    ).to(query.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights_before_softmax, key_states, value_states

class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        print("qwen2 head_dim: ",self.head_dim)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False, dtype=torch.float16)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False, dtype=torch.float16)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_output, attn_weights, key_states, value_states = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return dict(
            query_states=query_states,key_states=key_states,value_states=value_states,
            attn_output=attn_output, attn_weights=attn_weights
        )


def help_print(name:str, tensor:torch.Tensor):
    print(f"{name}:\n", f"shape: {tensor.shape}\n", tensor)


# test precision
@torch.no_grad()
def test():
    qwen2_attn = Qwen2Attention(config=my_llm_config, layer_idx=0)
    my_attn = BaseAttention(config=my_llm_config)

    hidden_states = torch.randn(size=(3, 12, my_llm_config.hidden_size), dtype=torch.float16)
    my_attn.load_state_dict(qwen2_attn.state_dict(), strict=True)

    qwen2_attn_output = qwen2_attn(hidden_states)
    my_attn_output = my_attn(hidden_states)
    # evaluate intermediate tensor
    # query, key, value states
    # help_print("qwen2_query_state",qwen2_attn_outputs['query_states'])
    # help_print("my_query_state",my_attn_outputs['query_states'])

    #NOTE method 1 to evaluate if two tensors are equal
    # query_equal:torch.Tensor = qwen2_attn_output['query_states']==my_attn_output['query_states']
    # key_equal:torch.Tensor = qwen2_attn_output['key_states']==my_attn_output['key_states']
    # value_equal:torch.Tensor = qwen2_attn_output['value_states']==my_attn_output['value_states']
    # print("query_states equal: ", torch.all(query_equal).item())
    # print("key_states equal: ", torch.all(key_equal).item())
    # print("value_states equal: ", torch.all(value_equal).item())

    #NOTE method 2 to evaluate if two tensors are equal within a slightly small margin of error
    query_equal = torch.allclose(qwen2_attn_output['query_states'], my_attn_output['query_states'])
    key_equal = torch.allclose(qwen2_attn_output['key_states'], my_attn_output['key_states'])
    value_equal = torch.allclose(qwen2_attn_output['value_states'], my_attn_output['value_states'])
    print("query_states equal: ", query_equal)
    print("key_states equal: ", key_equal)
    print("value_states equal: ", value_equal)

    # evaluate attn_weights: softmax(Q*K.T / dim**0.5)
    # attn_weights_equal: torch.Tensor = qwen2_attn_output['attn_weights'] == my_attn_output['attn_weights']
    # print("attn weights equal: ", torch.all(attn_weights_equal).item())
    attn_weights_equal = torch.allclose(qwen2_attn_output['attn_weights'],my_attn_output['attn_weights'])
    print("attn_weights_equal: ",attn_weights_equal)


    # print("qwen2_attn_outputs:\n",f"shape: {qwen2_attn_output.shape}\n",qwen2_attn_output)
    # print("my_attn_outputs:\n",f"shape:\n{my_attn_output.shape}\n",my_attn_output)
    attn_output_equal = torch.allclose(qwen2_attn_output['attn_output'], my_attn_output['attn_output'])
    print("attn_output equal: ", attn_output_equal)

if __name__ == '__main__':
    test()
