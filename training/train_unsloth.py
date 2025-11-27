#!/usr/bin/env python3
"""
DeepSeek OCR Fine-tuning with Unsloth for Turkish E-Invoice Extraction.
Using Unsloth for 1.4x faster training with 40% less VRAM.
"""

import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from PIL import Image
from trl import SFTTrainer, SFTConfig

# Suppress warnings
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

from unsloth import FastVisionModel
from transformers import AutoModel

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "unsloth/DeepSeek-OCR"
DATASET_DIR = "./dataset"
OUTPUT_DIR = "./checkpoints"
MAX_SEQ_LENGTH = 4096

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# LoRA configuration
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# ============================================================================
# Load Model with Unsloth
# ============================================================================

print("=" * 60)
print("DeepSeek OCR Fine-tuning with Unsloth")
print("Turkish E-Invoice Extraction")
print("=" * 60)

print("\n[1/5] Loading model with Unsloth...")

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=False,  # Use 16bit for better quality
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing="unsloth",  # For long context
)

print(f"Model loaded: {MODEL_NAME}")

# ============================================================================
# Setup LoRA
# ============================================================================

print("\n[2/5] Setting up LoRA adapters...")

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # Fine-tune vision encoder
    finetune_language_layers=True,  # Fine-tune language model
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

print(f"LoRA configured: r={LORA_R}, alpha={LORA_ALPHA}")

# ============================================================================
# Load Dataset
# ============================================================================

print("\n[3/5] Loading dataset...")

def load_jsonl(path):
    """Load JSONL file."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def prepare_sample(sample):
    """Prepare a single sample for training."""
    # Get image path(s)
    if "image" in sample:
        image_paths = [sample["image"]]
    else:
        image_paths = sample["images"]
    
    # Load images
    images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        else:
            print(f"Warning: Image not found: {img_path}")
    
    if not images:
        return None
    
    # Get conversation
    conversations = sample["conversations"]
    user_content = conversations[0]["content"]
    assistant_content = conversations[1]["content"]
    
    return {
        "images": images,
        "user_prompt": user_content,
        "assistant_response": assistant_content,
    }

# Load training data
train_samples = load_jsonl(os.path.join(DATASET_DIR, "train.jsonl"))
val_samples = load_jsonl(os.path.join(DATASET_DIR, "val.jsonl"))

print(f"Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples")

# ============================================================================
# Create Training Dataset
# ============================================================================

def create_dataset(samples):
    """Create HuggingFace dataset from samples."""
    processed = []
    for sample in samples:
        prepared = prepare_sample(sample)
        if prepared:
            processed.append(prepared)
    return processed

train_data = create_dataset(train_samples)
val_data = create_dataset(val_samples)

print(f"Prepared {len(train_data)} training samples, {len(val_data)} validation samples")

# ============================================================================
# Training with Unsloth
# ============================================================================

print("\n[4/5] Starting training...")

# Create training config
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    logging_steps=1,
    save_steps=50,
    save_total_limit=3,
    optim="adamw_8bit",
    seed=42,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
)

# Format data for training
def formatting_func(examples):
    """Format examples for training."""
    texts = []
    for user_prompt, assistant_response in zip(
        examples["user_prompt"], 
        examples["assistant_response"]
    ):
        # DeepSeek OCR format
        text = f"<|User|>{user_prompt}<|Assistant|>{assistant_response}<|end▁of▁sentence|>"
        texts.append(text)
    return {"text": texts}

# Convert to HuggingFace dataset
train_dataset = Dataset.from_list(train_data)
train_dataset = train_dataset.map(formatting_func, batched=True)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)

# Train
print("\nStarting training loop...")
trainer.train()

# ============================================================================
# Save Model
# ============================================================================

print("\n[5/5] Saving model...")

# Save LoRA adapters
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))

# Optionally merge and save full model
# model.save_pretrained_merged(os.path.join(OUTPUT_DIR, "merged"), tokenizer)

print(f"\nModel saved to {OUTPUT_DIR}")
print("=" * 60)
print("Training complete!")
print("=" * 60)
