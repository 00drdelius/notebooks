# position embeddings
implementing position embeddings layer in LLM and align the precision with transformers implementation.

test method:
```python
equal = torch.allclose(my_developed_output, transformers_developed_outputs)
if equal:
    print("all aligned")
else:
    print("not aligned")
```

current tested implementations:
1. [basic rotary position embeddings](./basic_rope.py)
