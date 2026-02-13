# feed-forward network
implementing feed-forward network layer in LLM and align the precision with transformers implementation.

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
1. first MoE application in LLM: https://arxiv.org/pdf/2101.03961


## current tested implementations
1. [Dense FFN, Multi-Layer Perceptron](./mlp.py)
2. [Sparse FFN, Mixture of Experts](./moe.py)