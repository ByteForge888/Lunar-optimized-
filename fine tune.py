from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
from datasets import load_dataset
dataset = load_dataset("json", data_files="lunar_analyzer.jsonl")
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(output_dir="./code_llama_lunar", per_device_train_batch_size=2, num_train_epochs=3, fp16=True)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])
trainer.train()
model.save_pretrained("./code_llama_lunar")