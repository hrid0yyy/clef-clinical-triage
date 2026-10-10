import csv
import json
import torch
from torch.utils.data import Dataset

def safe_encode_record(encode_record_fn, record, processor):
    """Safely invokes encode_record supporting all release signatures."""
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        return encode_record_fn(tokenizer, record, processor=processor)
    except TypeError:
        try:
            return encode_record_fn(record, processor)
        except TypeError:
            return encode_record_fn(record)

class SymptomDataset(Dataset):
    """
    Research-Grade Dataset loader for Symptom2Disease formatted for Clef Decision Models.
    Ensures 100% deterministic alignment between candidate options and target indices.
    """
    def __init__(self, csv_path, criteria_dict, question_instruction, encode_record_fn, processor, use_criteria=True):
        self.samples = []
        self.encode_record_fn = encode_record_fn
        self.processor = processor
        self.use_criteria = use_criteria
        
        # Enforce canonical, deterministic option order
        self.classes = sorted(list(criteria_dict.keys()))
        if use_criteria:
            self.criteria = {c: criteria_dict[c] for c in self.classes}
        else:
            self.criteria = {c: None for c in self.classes}
            
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.instruction = question_instruction
        
        # Read CSV
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"].strip()
                if label not in self.class_to_idx:
                    continue
                self.samples.append({
                    "original_id": row.get("original_id", ""),
                    "text": row["text"].strip(),
                    "label": label,
                    "target_idx": self.class_to_idx[label]
                })
                
        print(f"Loaded {len(self.samples)} samples from {csv_path} across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Formulate native Clef record schema
        record = {
            "state": item["text"],
            "questions": {
                "disease": {
                    "type": "choice",
                    "instructions": self.instruction,
                    "criteria": self.criteria
                }
            }
        }
        
        # Encode record
        encoded = safe_encode_record(self.encode_record_fn, record, self.processor)
        
        # Find exact index in encoded question options to prevent any misalignment
        target_idx = item["target_idx"]
        if hasattr(encoded, "questions") and len(encoded.questions) > 0:
            q = encoded.questions[0]
            if hasattr(q, "option_ids") and item["label"] in q.option_ids:
                target_idx = q.option_ids.index(item["label"])

        return {
            "encoded_record": encoded,
            "target": target_idx,
            "label": item["label"],
            "text": item["text"]
        }

def create_collate_fn(collate_records_fn, pad_token_id, device):
    """Creates a custom collator using Clef's native collate_records function."""
    def collate_fn(batch):
        encoded_list = [b["encoded_record"] for b in batch]
        targets = torch.tensor([b["target"] for b in batch], dtype=torch.long)
        labels = [b["label"] for b in batch]
        texts = [b["text"] for b in batch]
        
        # Collate encoded records into model tensor batch
        batch_inputs = collate_records_fn(encoded_list, pad_token_id, device)
        
        return {
            "inputs": batch_inputs,
            "targets": targets.to(device),
            "labels": labels,
            "texts": texts
        }
    return collate_fn
