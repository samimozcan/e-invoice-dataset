import os
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
from PIL import Image, ImageOps
from torchvision import transforms
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

# --- Helper Functions from DeepSeek-OCR ---

def load_image(image_path):
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        print(f"error loading image {image_path}: {e}")
        try:
            return Image.open(image_path)
        except:
            return None

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=9, image_size=640, use_thumbnail=False): # Changed min_num to 1 based on infer logic? No, infer uses default.
    # infer uses default min_num=2? No, let's check infer source.
    # infer calls dynamic_preprocess(image). Defaults are min_num=2, max_num=9.
    # But wait, if image is small, it might use 1x1.
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio

def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)
    return transform

class BasicImageTransform:
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = mean
        self.std = std
        transform_pipelines = [transforms.ToTensor()]
        normalize_t = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize_t is not None:
            transform_pipelines.append(normalize_t)
        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        return self.transform(x)

def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1 # Fallback
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2 # Fallback
    
    # DeepSeek might use specific IDs. 
    # In modeling code: bos_id = 0, eos_id = 1.
    # Let's trust the tokenizer unless we see explicit hardcoding.
    # The modeling code had: bos_id = 0, eos_id = 1.
    # I will use tokenizer values but fallback to 0/1 if None.
    
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]
    return t

# --- Dataset and Collator ---

class DeepSeekOCRDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DeepSeekOCRDataCollator:
    def __init__(self, tokenizer, image_size=640, base_size=1024, patch_size=16, downsample_ratio=4):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.base_size = base_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        self.image_token = '<image>'
        # self.image_token_id = 128815 # From modeling code
        # Ideally get from tokenizer
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        if self.image_token_id is None:
             self.image_token_id = 128815 # Fallback
             
    def __call__(self, batch):
        input_ids_batch = []
        labels_batch = []
        images_batch = []
        images_seq_mask_batch = []
        images_spatial_crop_batch = []
        
        for item in batch:
            prompt = item['prompt']
            response = item['response']
            image_paths = item['images']
            
            # Load images
            images = []
            for img_path in image_paths:
                img = load_image(img_path)
                if img:
                    images.append(img.convert("RGB"))
            
            # Prepare text
            # We want to train on response.
            # Format: Prompt + Response + EOS
            # We mask Prompt for labels.
            
            # Split prompt by <image> to insert image tokens
            text_splits = prompt.split(self.image_token)
            
            tokenized_str = []
            images_seq_mask = []
            images_list = []
            images_crop_list = []
            images_spatial_crop = []
            
            # Process prompt and images
            # Note: The loop assumes len(text_splits) == len(images) + 1
            # But if prompt ends with <image>, split might behave differently.
            # Usually prompt: <image> text <image> text ...
            
            # If prompt starts with <image>, text_splits[0] is empty.
            
            current_images_spatial_crop = []
            
            for i, text_sep in enumerate(text_splits):
                tokenized_sep = text_encode(self.tokenizer, text_sep, bos=False, eos=False)
                tokenized_str += tokenized_sep
                images_seq_mask += [False] * len(tokenized_sep)
                
                if i < len(images):
                    image = images[i]
                    
                    # Image processing logic from infer
                    # 1. Dynamic preprocess
                    images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=self.image_size)
                    
                    # 2. Global view
                    global_view = ImageOps.pad(image, (self.base_size, self.base_size),
                                            color=tuple(int(x * 255) for x in self.image_transform.mean))
                    images_list.append(self.image_transform(global_view).to(torch.bfloat16))
                    
                    width_crop_num, height_crop_num = crop_ratio
                    current_images_spatial_crop.append([width_crop_num, height_crop_num])
                    
                    # 3. Local views
                    if width_crop_num > 1 or height_crop_num > 1:
                        for crop_img in images_crop_raw:
                            images_crop_list.append(self.image_transform(crop_img).to(torch.bfloat16))
                            
                    # 4. Add image tokens
                    num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
                    num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
                    
                    tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
                    tokenized_image += [self.image_token_id]
                    
                    if width_crop_num > 1 or height_crop_num > 1:
                        tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                                    num_queries * height_crop_num)
                                    
                    tokenized_str += tokenized_image
                    images_seq_mask += [True] * len(tokenized_image)
            
            # Add BOS to start
            bos_id = 0 # From modeling code
            tokenized_str = [bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
            
            # Prompt processing done. Now add Response.
            # Response is text only.
            response_tokens = text_encode(self.tokenizer, response, bos=False, eos=True)
            
            input_ids = tokenized_str + response_tokens
            # Mask for images should be extended
            images_seq_mask += [False] * len(response_tokens)
            
            # Labels: -100 for prompt, response_tokens for response
            labels = [-100] * len(tokenized_str) + response_tokens
            
            input_ids_batch.append(torch.LongTensor(input_ids))
            labels_batch.append(torch.LongTensor(labels))
            images_seq_mask_batch.append(torch.tensor(images_seq_mask, dtype=torch.bool))
            
            # Stack images for this sample
            if len(images_list) > 0:
                images_ori = torch.stack(images_list, dim=0)
                images_spatial_crop_tensor = torch.tensor(current_images_spatial_crop, dtype=torch.long)
                if images_crop_list:
                    images_crop = torch.stack(images_crop_list, dim=0)
                else:
                    images_crop = torch.zeros((1, 3, self.base_size, self.base_size)).to(torch.bfloat16)
                
                # We need to return (images_crop, images_ori) tuple as expected by forward
                # But collator needs to batch them.
                # The model forward expects `images` as a list of tuples (one per sample in batch)
                # or a tensor if batching is handled differently.
                # Looking at forward: `images` type is `Optional[torch.FloatTensor]`.
                # But in `infer`, it passes `images=[(images_crop.cuda(), images_ori.cuda())]`.
                # And `images_spatial_crop` is passed.
                # In `forward`: `for image, crop_shape in zip(images, images_spatial_crop):`
                # So `images` should be a list of (crop, ori) tuples.
                
                images_batch.append((images_crop, images_ori))
                images_spatial_crop_batch.append(images_spatial_crop_tensor)
            else:
                # Handle no image case if needed
                pass

        # Pad input_ids and labels
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100)
        images_seq_mask_padded = torch.nn.utils.rnn.pad_sequence(images_seq_mask_batch, batch_first=True, padding_value=False)
        
        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "images": images_batch, # List of tuples
            "images_seq_mask": images_seq_mask_padded,
            "images_spatial_crop": images_spatial_crop_batch # List of tensors
        }

def train():
    model_name = "deepseek-ai/DeepSeek-OCR"
    output_dir = "deepseek_ocr_finetune"
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with trust_remote_code=True
    # We use AutoModelForCausalLM because it has the language modeling head.
    # The modeling code defines `DeepseekOCRForCausalLM`.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        _attn_implementation='flash_attention_2'
    )
    
    # Enable gradient checkpointing if needed
    model.gradient_checkpointing_enable()
    
    dataset = DeepSeekOCRDataset("dataset_multi_image.jsonl")
    collator = DeepSeekOCRDataCollator(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Small batch size for VLM
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        bf16=True, # Use bf16 as in inference code
        dataloader_pin_memory=False,
        remove_unused_columns=False, # Important for custom collator
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train()
