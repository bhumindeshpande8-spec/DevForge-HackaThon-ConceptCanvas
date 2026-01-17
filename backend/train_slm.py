import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# ---------- Load dataset ----------
dataset = load_dataset("json", data_files="slm_dataset_hf.jsonl", split="train")

# ---------- Load model & tokenizer ----------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT2 tokenizer may not have pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------- Preprocess dataset ----------
def preprocess(batch):
    inputs = batch["input_text"]
    targets = batch["target_text"]
    
    # Tokenize inputs and labels with same max_length
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)["input_ids"]
    
    # Ensure labels match input length
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# ---------- Training arguments (CPU friendly) ----------
training_args = TrainingArguments(
    output_dir="./trained_slm",
    num_train_epochs=3,              # reduce to 1 for quicker test
    per_device_train_batch_size=1,   # 1 for CPU
    logging_steps=20,
    save_steps=50,
    save_total_limit=2,
    fp16=False,                      # CPU only, no float16
)

# ---------- Trainer ----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# ---------- Train ----------
trainer.train()

# ---------- Save model ----------
model.save_pretrained("trained_slm")
tokenizer.save_pretrained("trained_slm")

print("✅ Training complete. Model saved to ./trained_slm")
