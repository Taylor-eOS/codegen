import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text = input("Input:")
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

