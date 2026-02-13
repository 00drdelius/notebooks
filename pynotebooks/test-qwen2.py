from typing import cast
import torch
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer

model_path='/models/Qwen/Qwen2.5-3B-Instruct'


model = Qwen2ForCausalLM.from_pretrained(model_path,device_map="cuda:0").eval()
tokenizer:Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_path)

prompt = [
    "Hey, are you conscious? Can you talk to me?",
    "What's the capital city of China?"
]
inputs = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left")
inputs=inputs.to('cuda:0')
print(inputs)
input_ids = cast(torch.Tensor, inputs.input_ids)
prefix_len = input_ids.shape[-1]
# Generate
generate_ids = model.generate(input_ids, max_length=30)
generate_ids=generate_ids[:, prefix_len:]
result=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(result)
