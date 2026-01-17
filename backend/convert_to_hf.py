import json

with open("slm_dataset.json") as f:
    data = json.load(f)

with open("slm_dataset_hf.jsonl", "w") as f:
    for item in data:
        line = {
            "input_text": json.dumps(item["input"]),
            "target_text": item["output"]
        }
        f.write(json.dumps(line) + "\n")

print("Dataset converted to slm_dataset_hf.jsonl for Hugging Face")
