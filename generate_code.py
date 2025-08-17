from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "meta-llama/CodeLlama-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)