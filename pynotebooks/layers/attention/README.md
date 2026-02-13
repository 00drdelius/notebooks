# attention
implementing attention layer in LLM and align the precision with transformers implementation.

## precision test method
```python
equal = torch.allclose(my_developed_output, transformers_developed_outputs)
if equal:
    print("all aligned")
else:
    print("not aligned")
```

## NOTE
You may refer more to the following url for details:
1. original paper: https://arxiv.org/pdf/2305.13245

## current tested implementations
1. [Grouped Query Attention](./basic_attention.py)
