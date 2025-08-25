import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

BASE = "mistralai/Mistral-7B-v0.1"  # choose a base model you can access
LOAD_IN_4BIT = True  # set False to force CPU if needed

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if LOAD_IN_4BIT:
    model = AutoModelForCausalLM.from_pretrained(BASE, load_in_4bit=True, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(BASE)

lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, lora_cfg)

ds = load_dataset("json", data_files="instructions/train.jsonl")["train"]

def format_example(ex):
    sys = "You are a dental clinical drafting assistant. Output VALID JSON exactly matching the schema. If missing info, set safety.needs_human_review=true."
    retrieved = ex.get("kb_snippets","")
    return f"SYSTEM:\n{sys}\n\nRETRIEVED:\n{retrieved}\n\nINPUT_JSON:\n{ex['prompt']}\n\nOUTPUT_JSON:\n"

def tokenize(ex):
    text = format_example(ex) + ex["response"]
    out = tokenizer(text, truncation=True, max_length=2048)
    out["labels"] = out["input_ids"].copy()
    return out

tok = ds.map(tokenize, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="adapters/dental-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    bf16=True if LOAD_IN_4BIT else False,
    logging_steps=10,
    save_strategy="epoch"
)

Trainer(model=model, args=args, train_dataset=tok).train()
model.save_pretrained("adapters/dental-lora")
print("Saved LoRA adapter to adapters/dental-lora")
