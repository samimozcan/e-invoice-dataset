#!/usr/bin/env python3
"""
DeepSeek OCR Fine-tuning Script for Turkish E-Invoice Extraction.
Supports full fine-tuning and LoRA-based parameter-efficient fine-tuning.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import numpy as np

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    get_cosine_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint

# Add the parent directory to path for importing DeepSeek OCR model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_ocr_code.modeling_deepseekocr import (
    DeepseekOCRForCausalLM,
    DeepseekOCRConfig,
    format_messages,
    text_encode,
    dynamic_preprocess,
    BasicImageTransform,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="./deepseek_ocr_code")
    freeze_vision_encoder: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class DataArguments:
    dataset_path: str = field(default="./dataset")
    max_length: int = field(default=4096)
    image_size: int = field(default=1024)
    patch_size: int = field(default=640)
    use_dynamic_preprocess: bool = field(default=True)
    min_patches: int = field(default=2)
    max_patches: int = field(default=9)


class InvoiceDataset(Dataset):
    """Dataset for Turkish e-invoice images and structured outputs."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_size: int = 1024,
        patch_size: int = 640,
        max_length: int = 4096,
        use_dynamic_preprocess: bool = True,
        min_patches: int = 2,
        max_patches: int = 9,
        is_training: bool = True,
    ):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_length = max_length
        self.use_dynamic_preprocess = use_dynamic_preprocess
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.is_training = is_training
        
        # Load data
        self.samples = self._load_data(data_path)
        
        # Image transform
        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True
        )
        
        # Token IDs
        self.image_token_id = 128815
        self.bos_id = 0
        self.eos_id = 1
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load JSONL dataset file."""
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _process_image(self, image_path: str) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Process image with dynamic cropping."""
        # Load and convert image
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        
        images_list = []
        images_crop_list = []
        
        if self.use_dynamic_preprocess:
            w, h = image.size
            
            if w <= self.patch_size and h <= self.patch_size:
                crop_ratio = [1, 1]
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image,
                    min_num=self.min_patches,
                    max_num=self.max_patches,
                    image_size=self.patch_size
                )
                
                # Process cropped images
                for crop_img in images_crop_raw:
                    images_crop_list.append(
                        self.image_transform(crop_img).to(torch.bfloat16)
                    )
            
            # Process global view
            global_view = ImageOps.pad(
                image,
                (self.image_size, self.image_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(torch.bfloat16))
        else:
            # Simple resize
            image = image.resize((self.image_size, self.image_size))
            images_list.append(self.image_transform(image).to(torch.bfloat16))
            crop_ratio = [1, 1]
        
        # Stack tensors
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, self.patch_size, self.patch_size))
        
        images_ori = torch.stack(images_list, dim=0)
        
        return images_crop, images_ori, crop_ratio
    
    def _tokenize_conversation(
        self,
        conversations: List[Dict[str, str]],
        crop_ratio: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize conversation with image tokens."""
        patch_size = 16
        downsample_ratio = 4
        
        # Format conversation
        prompt = format_messages(
            conversations=conversations,
            sft_format='plain',
            system_prompt=''
        )
        
        # Split by image token
        image_token = '<image>'
        text_splits = prompt.split(image_token)
        
        tokenized_str = []
        images_seq_mask = []
        
        width_crop_num, height_crop_num = crop_ratio
        
        # Process first text part
        if text_splits:
            first_tokens = text_encode(self.tokenizer, text_splits[0], bos=False, eos=False)
            tokenized_str.extend(first_tokens)
            images_seq_mask.extend([False] * len(first_tokens))
        
        # Add image tokens
        num_queries = (self.patch_size // patch_size) // downsample_ratio
        num_queries_base = (self.image_size // patch_size) // downsample_ratio
        
        # Global view tokens
        tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
        tokenized_image += [self.image_token_id]
        
        # Local view tokens (if using dynamic preprocessing with multiple patches)
        if width_crop_num > 1 or height_crop_num > 1:
            tokenized_image += (
                [self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]
            ) * (num_queries * height_crop_num)
        
        tokenized_str.extend(tokenized_image)
        images_seq_mask.extend([True] * len(tokenized_image))
        
        # Process remaining text parts
        for text_part in text_splits[1:]:
            tokens = text_encode(self.tokenizer, text_part, bos=False, eos=False)
            tokenized_str.extend(tokens)
            images_seq_mask.extend([False] * len(tokens))
        
        # Add BOS token at the beginning
        tokenized_str = [self.bos_id] + tokenized_str
        images_seq_mask = [False] + images_seq_mask
        
        # Add EOS token at the end for training
        if self.is_training:
            tokenized_str.append(self.eos_id)
            images_seq_mask.append(False)
        
        # Truncate if necessary
        if len(tokenized_str) > self.max_length:
            tokenized_str = tokenized_str[:self.max_length]
            images_seq_mask = images_seq_mask[:self.max_length]
        
        input_ids = torch.tensor(tokenized_str, dtype=torch.long)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)
        
        # Create labels (shift input_ids for next token prediction)
        labels = input_ids.clone()
        
        # Mask image tokens and padding in labels
        labels[images_seq_mask] = -100
        
        return input_ids, labels, images_seq_mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Process image
        image_path = sample["image"]
        images_crop, images_ori, crop_ratio = self._process_image(image_path)
        
        # Tokenize conversation
        input_ids, labels, images_seq_mask = self._tokenize_conversation(
            sample["conversations"],
            crop_ratio
        )
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": (images_crop, images_ori),
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": torch.tensor(crop_ratio, dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences."""
    # Find max length in batch
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    # Pad sequences
    input_ids = []
    attention_mask = []
    labels = []
    images_seq_mask = []
    images = []
    images_spatial_crop = []
    
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        
        # Pad input_ids
        padded_input_ids = torch.cat([
            item["input_ids"],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        input_ids.append(padded_input_ids)
        
        # Pad attention_mask
        padded_attention = torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        attention_mask.append(padded_attention)
        
        # Pad labels with -100 (ignore index)
        padded_labels = torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        labels.append(padded_labels)
        
        # Pad images_seq_mask
        padded_mask = torch.cat([
            item["images_seq_mask"],
            torch.zeros(pad_len, dtype=torch.bool)
        ])
        images_seq_mask.append(padded_mask)
        
        images.append(item["images"])
        images_spatial_crop.append(item["images_spatial_crop"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "images": images,
        "images_seq_mask": torch.stack(images_seq_mask),
        "images_spatial_crop": torch.stack(images_spatial_crop),
    }


class DeepSeekOCRTrainer(Trainer):
    """Custom trainer for DeepSeek OCR."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute the loss for DeepSeek OCR."""
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            images=inputs["images"],
            images_seq_mask=inputs["images_seq_mask"],
            images_spatial_crop=inputs["images_spatial_crop"],
        )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def setup_lora(model, model_args: ModelArguments):
    """Setup LoRA for parameter-efficient fine-tuning."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    except ImportError:
        logger.warning("PEFT not installed. Skipping LoRA setup.")
        return model


def freeze_model_parts(model, model_args: ModelArguments):
    """Freeze specified parts of the model."""
    if model_args.freeze_vision_encoder:
        for name, param in model.named_parameters():
            if "sam_model" in name or "vision_model" in name:
                param.requires_grad = False
        logger.info("Froze vision encoder parameters")
    
    if model_args.freeze_llm:
        for name, param in model.named_parameters():
            if "model.layers" in name or "model.embed_tokens" in name:
                param.requires_grad = False
        logger.info("Froze LLM parameters")


def train():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./training/config.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup arguments
    model_args = ModelArguments(
        model_name_or_path=config.get("model_name_or_path", "./deepseek_ocr_code"),
        **config.get("model", {})
    )
    
    data_args = DataArguments(
        dataset_path=config.get("dataset_path", "./dataset"),
        **config.get("data", {})
    )
    
    training_config = config.get("training", {})
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "./checkpoints"),
        num_train_epochs=training_config.get("num_train_epochs", 10),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", True),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 100),
        eval_steps=training_config.get("eval_steps", 50),
        save_total_limit=training_config.get("save_total_limit", 3),
        eval_strategy=training_config.get("evaluation_strategy", "steps"),
        save_strategy=training_config.get("save_strategy", "steps"),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        report_to=training_config.get("report_to", "tensorboard"),
        local_rank=args.local_rank,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    
    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    model = DeepseekOCRForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Freeze model parts if specified
    freeze_model_parts(model, model_args)
    
    # Setup LoRA if enabled
    if model_args.use_lora:
        model = setup_lora(model, model_args)
    
    # Load datasets
    train_dataset = InvoiceDataset(
        data_path=os.path.join(data_args.dataset_path, "train.jsonl"),
        tokenizer=tokenizer,
        image_size=data_args.image_size,
        patch_size=data_args.patch_size,
        max_length=data_args.max_length,
        use_dynamic_preprocess=data_args.use_dynamic_preprocess,
        min_patches=data_args.min_patches,
        max_patches=data_args.max_patches,
        is_training=True,
    )
    
    eval_dataset = InvoiceDataset(
        data_path=os.path.join(data_args.dataset_path, "val.jsonl"),
        tokenizer=tokenizer,
        image_size=data_args.image_size,
        patch_size=data_args.patch_size,
        max_length=data_args.max_length,
        use_dynamic_preprocess=data_args.use_dynamic_preprocess,
        min_patches=data_args.min_patches,
        max_patches=data_args.max_patches,
        is_training=False,
    )
    
    # Initialize trainer
    trainer = DeepSeekOCRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )
    
    # Check for existing checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save final model
    trainer.save_model()
    logger.info(f"Model saved to {training_args.output_dir}")
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
