import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock, Qwen3MoeMLP

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



class BaseMoe(nn.Module):
    def __init__(self,llm_config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = llm_config.num_experts
        self.num_experts_per_tok = llm_config.num_experts_per_tok
        
        self.gate = nn.Linear(llm_config.hidden_size, llm_config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Qwen3MoeMLP(llm_config, intermediate_size=llm_config.moe_intermediate_size)
            for i in range(self.num_experts)
        ])
    
    def forward(self, hidden_states:torch.Tensor):
        bsz, seq_len, hidden_size = hidden_states.shape

        # concat all batch of tokens into one tensor
        hidden_states = hidden_states.view(bsz*seq_len, hidden_size)

        router_logits:torch.Tensor = self.gate(hidden_states) # [bsz*seq_len, num_experts]
        router_weights = F.softmax(router_logits, dim=-1) # normalize along the last dim

        # topk returns two tensors with shape in [bsz*seq_len, num_experts_per_tok].
        # First is values, second is every value's relating index corresponding to the input_tensor 
        topk_weights, topk_indices = torch.topk(router_weights, k=self.num_experts_per_tok, dim=-1)

        # `one_hot` extends the last dim of the input tensors to one hot tensors,
        # and every element in the given tensors stands for the index of 1 in the generated one hot tensor.
        # expert_mask before permute: [bsz*seq_len, num_experts_per_tok, num_experts]
        # after permute: [num_experts, num_experts_per_tok, bsz*seq_len]
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 1, 0)

        # create final hidden states
        output_hidden_states = torch.zeros(size=(bsz*seq_len, hidden_size)).to(
            dtype=hidden_states.dtype, device=hidden_states.device)

        # sum all [num_experts_per_tok, bsz*seq_len] along the dim of num_experts
        # and use `nonzero` to get the indices of the non-zero element,
        # for removing the experts that are not hit.
        # see: https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html#torch-nonzero for more details.
        expert_hit = (torch.sum(expert_mask, dim=(-1,-2)) > 0).nonzero()

        for expert_idx in expert_hit:
            expert:Qwen3MoeMLP = self.experts[expert_idx]

            # `torch.where(input_tensor)` is equal to `torch.nonzero(input_tensor, as_tuple=True)`
            # see: https://docs.pytorch.org/docs/stable/generated/torch.where.html#torch-where for more details
            # Returns a tuple that the first element is row indices of the nonzero-element in input_tensor, 
            # the second element is col indices of nonzero-element in input_tensor 
            row_indices, col_indices = torch.where(expert_mask[expert_idx].squeeze(0))
            
            # col_indices tensor indicates the selected tokens, so we choose the current_states
            # by selecting the corresponding indices from hidden_states
            current_states = hidden_states[col_indices]
            # forward by expert and element-wise product the weights
            # topk_weights: [bsz*seq_len, num_experts_per_tok],
            # col_indices indicates the selected tokens; row_indices indicates the selected expert
            # so we select the corresponding weights from topk_weights by [col_indices, row_indices]
            current_states = expert(current_states) * topk_weights[col_indices, row_indices, None] #NOTE topk_weights must expand the last dimension by None for boradcast
            output_hidden_states.index_add_(dim=0, index=col_indices, source=current_states)
        
        output_hidden_states = output_hidden_states.reshape(bsz, seq_len, hidden_size)

        return output_hidden_states, router_logits

@torch.no_grad()
def test():
    torch.set_default_device("cpu")
    my_moe = BaseMoe(my_llm_moe_config)
    qwen3_moe = Qwen3MoeSparseMoeBlock(my_llm_moe_config)

    bsz, seq_len, hidden_size = 3, 14, my_llm_moe_config.hidden_size
    hidden_states = torch.randn(size=(bsz, seq_len, hidden_size))
    
    qwen3_result = qwen3_moe(hidden_states)[0]
    my_result = my_moe(hidden_states)

    print("[precision aligned] ", torch.allclose(my_result, qwen3_result))

test()