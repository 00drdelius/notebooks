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