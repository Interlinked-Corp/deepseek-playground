# deepseek_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
import json
import pandas as pd

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset
data_list = []
with open("playground/disaster_data_modified.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data_list.append(json.loads(line.strip()))

df = pd.DataFrame(data_list)
dataset = Dataset.from_pandas(df)
dataset = DatasetDict({"train": dataset})

print(f"Dataset size: {len(dataset['train'])} examples")

# Tokenize with consistent max_length
def tokenize_function(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    combined = [f"{instr}\n\nAnalysis:\n{out}" for instr, out in zip(instructions, outputs)]
    encodings = tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=768,
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()
    for i in range(len(instructions)):
        input_len = len(tokenizer.encode(instructions[i], add_special_tokens=False)) + len(tokenizer.encode("\n\nAnalysis:\n", add_special_tokens=False))
        labels[i, :input_len] = -100
    return {
        "input_ids": input_ids.squeeze(),
        "attention_mask": attention_mask.squeeze(),
        "labels": labels.squeeze()
    }

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["instruction", "output"]
)

# Training setup
training_args = TrainingArguments(
    output_dir="./deepseek_lora_output_training_args",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,  
    learning_rate=1e-5,  
    fp16=True,
    logging_steps=1,
    save_steps=5,
    save_total_limit=2,
    remove_unused_columns=False,
    optim="adamw_torch"
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f"Step loss: {loss.item()}")
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

total_samples = len(tokenized_dataset["train"])
effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
total_steps = (total_samples // effective_batch_size) * training_args.num_train_epochs
print(f"Total training steps: {total_steps}") # 이제 100 단계가 될 것입니다 (10샘플 / 4배치 * 10 에포크)

# Fine-tune
trainer.train()

# Save
model.save_pretrained("./deepseek_lora_output")
tokenizer.save_pretrained("./deepseek_lora_output")