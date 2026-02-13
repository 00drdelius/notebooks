# position embeddings
implementing position embeddings layer in LLM and align the precision with transformers implementation.

## precision test method
```python
equal = torch.allclose(my_developed_output, transformers_developed_outputs)
if equal:
    print("all aligned")
else:
    print("not aligned")
```

## NOTE
You may refer more to the following url for RoPE math details:
1. original paper: https://arxiv.org/pdf/2104.09864
2. improved implementation: https://nn.labml.ai/transformers/rope/index.html
3. different implementation styles brief: https://anyinlover.github.io/blog/deep_learning/deepseek_rope

## current tested implementations
1. [basic rotary position embeddings](./basic_rope.py)

