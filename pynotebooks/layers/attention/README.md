# attention
implementing attention layer in LLM and align the precision with transformers implementation.

test method:
```python
equal = torch.allclose(my_developed_output, transformers_developed_outputs)
if equal:
    print("all aligned")
else:
    print("not aligned")
```

current tested implementations:
1. [Grouped Query Attention](./basic_attention.py)
