#!/usr/bin/env python3
"""
Dataset preparation script for DeepSeek OCR fine-tuning on Turkish e-invoices.
Converts JSON annotations and images to training format.

Supports:
1. Single-page invoices: 1 image → 1 invoice
2. Multi-page single invoice: N images → 1 invoice (combined)
3. Multi-invoice PDFs: N images → N invoices (each page = 1 invoice)
4. Multi-page multi-invoice: All images combined → All invoices
"""

import os
import json
import glob
import re
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
from dataclasses import dataclass, asdict
import random

# Response schema for invoice extraction
INVOICE_SCHEMA = {
    "invoice_currency": "str or null",
    "invoice_customer_address": "str or null",
    "invoice_customer_country": "str or null",
    "invoice_customer_name": "str or null",
    "invoice_date": "str or null",
    "invoice_delivery_term": "str or null",
    "invoice_id": "str or null",
    "invoice_payment_term": "str or null",
    "invoice_po_number": "str or null",
    "invoice_shipment_country_of_origin": "str or null",
    "invoice_supplier_address": "str or null",
    "invoice_supplier_country": "str or null",
    "invoice_supplier_name": "str or null",
    "invoice_supplier_vkn": "str or null",
    "invoice_total_amount": "str or null",
    "invoice_total_package_quantity": "str or null",
    "invoice_total_quantity": "str or null",
    "invoice_total_gross_weight": "str or null",
    "invoice_total_net_weight": "str or null",
    "items": [
        {
            "invoice_item_commodity_code": "str or null",
            "invoice_item_country_of_origin": "str or null",
            "invoice_item_description": "str or null",
            "invoice_item_no": "str or null",
            "invoice_item_package_quantity": "str or null",
            "invoice_item_product_id": "str or null",
            "invoice_item_quantity": "str or null",
            "invoice_item_total_amount": "str or null",
            "invoice_item_unit_price": "str or null",
            "invoice_item_unit_type": "str or null"
        }
    ]
}

SYSTEM_PROMPT = """You are an expert document understanding AI specialized in extracting structured information from Turkish e-invoices (e-Fatura). 
Your task is to analyze invoice images and extract all relevant information into a structured JSON format.
Be precise and extract exact values as they appear in the document. If a field is not present, use null."""

# Single image prompt
USER_PROMPT_SINGLE = """<image>
Extract all information from this Turkish e-invoice image and return a structured JSON response following this schema:

```json
{schema}
```

Extract all invoice details including header information and all line items. Return only valid JSON."""

# Multi-image prompt (for multi-page documents)
USER_PROMPT_MULTI = """{image_tokens}
Extract all information from these Turkish e-invoice pages and return a structured JSON array with all invoices found. Each invoice should follow this schema:

```json
{schema}
```

Extract all invoice details including header information and all line items from all pages. Return a JSON array of invoices."""


@dataclass
class TrainingSample:
    """A single training sample for DeepSeek OCR."""
    images: List[str]  # Can be single or multiple images
    conversations: List[Dict[str, str]]
    
    def to_dict(self) -> Dict:
        # For single image, use "image" key; for multiple, use "images" key
        if len(self.images) == 1:
            return {
                "image": self.images[0],
                "conversations": self.conversations
            }
        else:
            return {
                "images": self.images,
                "conversations": self.conversations
            }


def extract_page_number(filename: str) -> int:
    """Extract page number from filename like 'xxx_page_1.jpg'."""
    match = re.search(r'_page_(\d+)\.jpg$', filename)
    if match:
        return int(match.group(1))
    return 0


def find_matching_images(pdf_name: str, images_dir: str) -> List[str]:
    """Find all images that belong to a PDF document, sorted by page number."""
    # Remove .json extension if present
    base_name = pdf_name.replace('.json', '')
    
    # Look for images with pattern: {base_name}_page_*.jpg
    pattern = os.path.join(images_dir, f"{base_name}_page_*.jpg")
    images = glob.glob(pattern)
    
    if not images:
        # Try without extension matching
        pattern = os.path.join(images_dir, f"{base_name}*page*.jpg")
        images = glob.glob(pattern)
    
    # Sort by page number
    images = sorted(images, key=extract_page_number)
    
    return images


def load_json_annotation(json_path: str) -> Dict[str, Any]:
    """Load JSON annotation file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_invoice_data(invoice: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize invoice data to ensure consistent types."""
    normalized = {}
    
    # Copy header fields, converting to string or null
    for key in INVOICE_SCHEMA.keys():
        if key == "items":
            continue
        value = invoice.get(key)
        if value is not None:
            normalized[key] = str(value) if not isinstance(value, str) else value
        else:
            normalized[key] = None
    
    # Process items
    normalized["items"] = []
    for item in invoice.get("items", []):
        normalized_item = {}
        for item_key in INVOICE_SCHEMA["items"][0].keys():
            value = item.get(item_key)
            if value is not None:
                normalized_item[item_key] = str(value) if not isinstance(value, str) else value
            else:
                normalized_item[item_key] = None
        normalized["items"].append(normalized_item)
    
    return normalized


def create_conversation_single(invoice_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create conversation format for single-image training."""
    schema_str = json.dumps(INVOICE_SCHEMA, indent=2, ensure_ascii=False)
    user_content = USER_PROMPT_SINGLE.format(schema=schema_str)
    
    # Format response as clean JSON
    response_json = json.dumps(invoice_data, indent=2, ensure_ascii=False)
    
    return [
        {"role": "<|User|>", "content": user_content},
        {"role": "<|Assistant|>", "content": response_json}
    ]


def create_conversation_multi(invoices: List[Dict[str, Any]], num_images: int) -> List[Dict[str, str]]:
    """Create conversation format for multi-image training."""
    schema_str = json.dumps(INVOICE_SCHEMA, indent=2, ensure_ascii=False)
    
    # Create image tokens for all pages
    image_tokens = "\n".join(["<image>"] * num_images)
    user_content = USER_PROMPT_MULTI.format(schema=schema_str, image_tokens=image_tokens)
    
    # Format response as JSON array
    response_json = json.dumps(invoices, indent=2, ensure_ascii=False)
    
    return [
        {"role": "<|User|>", "content": user_content},
        {"role": "<|Assistant|>", "content": response_json}
    ]


def determine_document_type(num_images: int, num_invoices: int) -> str:
    """
    Determine the type of document based on image and invoice counts.
    
    Returns:
        - "single": 1 image, 1 invoice
        - "multi_page_single": N images, 1 invoice (multi-page single invoice)
        - "multi_invoice_per_page": N images, N invoices (1 invoice per page)
        - "multi_page_multi": N images, M invoices where N != M
    """
    if num_images == 1 and num_invoices == 1:
        return "single"
    elif num_images > 1 and num_invoices == 1:
        return "multi_page_single"
    elif num_images == num_invoices:
        return "multi_invoice_per_page"
    else:
        return "multi_page_multi"


def get_unique_invoices(invoices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get unique invoices by merging duplicates based on invoice_number.
    If multiple invoice objects have the same invoice_number (or all are None),
    they are considered parts of the same invoice and should be merged.
    """
    if not invoices:
        return []
    
    # Group by invoice_number
    invoice_groups = {}
    for inv in invoices:
        inv_num = inv.get("invoice_number") or "NO_NUMBER"
        if inv_num not in invoice_groups:
            invoice_groups[inv_num] = []
        invoice_groups[inv_num].append(inv)
    
    unique_invoices = []
    for inv_num, group in invoice_groups.items():
        if len(group) == 1:
            unique_invoices.append(group[0])
        else:
            # Merge multiple objects into one - combine items
            merged = group[0].copy()
            all_items = []
            for inv in group:
                all_items.extend(inv.get("items", []))
            merged["items"] = all_items
            unique_invoices.append(merged)
    
    return unique_invoices


def create_training_samples(
    output_dir: str,
    images_dir: str,
    dataset_output_dir: str,
    mode: str = "auto"
) -> List[Dict]:
    """
    Create training samples from annotations and images.
    
    Args:
        output_dir: Directory containing JSON annotations
        images_dir: Directory containing images
        dataset_output_dir: Output directory for dataset
        mode: 
            - "auto": Automatically determine best approach per document
            - "per_page": Create one sample per page (1 image = 1 invoice)
            - "combined": Create one sample with all pages combined
    """
    samples = []
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    print(f"Found {len(json_files)} JSON annotation files")
    
    for json_path in json_files:
        json_name = os.path.basename(json_path)
        
        # Find matching images
        images = find_matching_images(json_name, images_dir)
        
        if not images:
            print(f"Warning: No images found for {json_name}")
            continue
        
        # Load annotation
        annotation = load_json_annotation(json_path)
        invoices = annotation.get("data", [])
        
        if not invoices:
            print(f"Warning: No invoice data in {json_name}")
            continue
        
        # Get unique invoices (merge duplicates with same invoice_number)
        unique_invoices = get_unique_invoices(invoices)
        
        num_images = len(images)
        num_invoices = len(unique_invoices)
        raw_invoice_count = len(invoices)
        doc_type = determine_document_type(num_images, num_invoices)
        
        if raw_invoice_count != num_invoices:
            print(f"Processing {json_name}: {num_images} images, {raw_invoice_count} raw -> {num_invoices} unique invoices ({doc_type})")
        else:
            print(f"Processing {json_name}: {num_images} images, {num_invoices} invoices ({doc_type})")
        
        # Normalize all unique invoices
        normalized_invoices = [normalize_invoice_data(inv) for inv in unique_invoices]
        
        if doc_type == "single":
            # Single page, single invoice: 1 sample
            sample = TrainingSample(
                images=[images[0]],
                conversations=create_conversation_single(normalized_invoices[0])
            )
            samples.append(sample.to_dict())
            
        elif doc_type == "multi_page_single":
            # Multi-page single invoice: combine all pages for one invoice
            # Create ONE sample with all images
            sample = TrainingSample(
                images=images,
                conversations=create_conversation_multi(normalized_invoices, num_images)
            )
            samples.append(sample.to_dict())
            
        elif doc_type == "multi_invoice_per_page":
            # Each page corresponds to one invoice: create per-page samples
            for img_path, invoice in zip(images, normalized_invoices):
                sample = TrainingSample(
                    images=[img_path],
                    conversations=create_conversation_single(invoice)
                )
                samples.append(sample.to_dict())
                
        else:  # multi_page_multi
            # Complex case: multiple invoices across multiple pages
            # Create ONE combined sample with all pages and all invoices
            # We cannot reliably map individual pages to invoices, so combine all
            sample = TrainingSample(
                images=images,
                conversations=create_conversation_multi(normalized_invoices, num_images)
            )
            samples.append(sample.to_dict())
    
    return samples


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """Split dataset into train/val/test sets."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:]
    }


def save_dataset(
    splits: Dict[str, List[Dict]],
    output_dir: str
):
    """Save dataset splits to JSONL files."""
    os.makedirs(output_dir, exist_ok=True)
    
    single_image_count = 0
    multi_image_count = 0
    
    for split_name, samples in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
                # Count sample types (check for both "image" and "images" keys)
                if "image" in sample:
                    single_image_count += 1
                elif "images" in sample:
                    multi_image_count += 1
                else:
                    single_image_count += 1  # fallback
        
        print(f"Saved {len(samples)} samples to {output_path}")
    
    print(f"\nSample statistics:")
    print(f"  - Single-image samples: {single_image_count}")
    print(f"  - Multi-image samples: {multi_image_count}")
    
    # Save schema file for reference
    schema_path = os.path.join(output_dir, "schema.json")
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(INVOICE_SCHEMA, f, indent=2, ensure_ascii=False)
    
    # Save dataset info
    info = {
        "total_samples": sum(len(s) for s in splits.values()),
        "splits": {k: len(v) for k, v in splits.items()},
        "single_image_samples": single_image_count,
        "multi_image_samples": multi_image_count,
        "schema": INVOICE_SCHEMA
    }
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for DeepSeek OCR training")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory containing JSON annotations")
    parser.add_argument("--images-dir", type=str, default="./images",
                        help="Directory containing invoice images")
    parser.add_argument("--dataset-dir", type=str, default="./dataset",
                        help="Output directory for processed dataset")
    parser.add_argument("--train-ratio", type=float, default=0.85,
                        help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.10,
                        help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.05,
                        help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "per_page", "combined"],
                        help="How to handle multi-page documents")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("DeepSeek OCR Dataset Preparation")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    
    # Create training samples
    samples = create_training_samples(
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        dataset_output_dir=args.dataset_dir,
        mode=args.mode
    )
    
    print(f"\nTotal samples created: {len(samples)}")
    
    # Split dataset
    splits = split_dataset(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Save dataset
    save_dataset(splits, args.dataset_dir)
    
    print("\n" + "=" * 50)
    print("Dataset preparation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
