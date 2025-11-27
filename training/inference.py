#!/usr/bin/env python3
"""
Inference script for trained DeepSeek OCR model on Turkish e-invoices.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_ocr_code.modeling_deepseekocr import DeepseekOCRForCausalLM


# Response schema for reference
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

USER_PROMPT = """<image>
Extract all information from this Turkish e-invoice image and return a structured JSON response following this schema:

```json
{schema}
```

Extract all invoice details including header information and all line items. Return only valid JSON."""


class InvoiceExtractor:
    """Extract structured data from Turkish e-invoice images."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        base_size: int = 1024,
        image_size: int = 640,
    ):
        self.device = device
        self.base_size = base_size
        self.image_size = image_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # Load model
        self.model = DeepseekOCRForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        # Check for LoRA adapter
        lora_path = os.path.join(model_path, "adapter_model.bin")
        if os.path.exists(lora_path):
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, model_path)
                print("Loaded LoRA adapter")
            except ImportError:
                print("PEFT not installed, skipping LoRA adapter loading")
    
    def extract(
        self,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract invoice data from an image.
        
        Args:
            image_path: Path to the invoice image
            output_path: Optional path to save extraction results
            
        Returns:
            Extracted invoice data as dictionary
        """
        schema_str = json.dumps(INVOICE_SCHEMA, indent=2, ensure_ascii=False)
        prompt = USER_PROMPT.format(schema=schema_str)
        
        # Run inference
        output = self.model.infer(
            tokenizer=self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_path or "./inference_output",
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=True,
            eval_mode=True,
        )
        
        # Parse JSON output
        try:
            # Clean up output
            output = output.strip()
            if output.startswith("```json"):
                output = output[7:]
            if output.startswith("```"):
                output = output[3:]
            if output.endswith("```"):
                output = output[:-3]
            
            result = json.loads(output)
        except json.JSONDecodeError:
            result = {"raw_output": output, "error": "Failed to parse JSON"}
        
        # Save results if output path provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            result_path = os.path.join(output_path, "extraction_result.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
    
    def batch_extract(
        self,
        image_paths: list,
        output_dir: str,
    ) -> list:
        """
        Extract invoice data from multiple images.
        
        Args:
            image_paths: List of paths to invoice images
            output_dir: Directory to save extraction results
            
        Returns:
            List of extraction results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.extract(
                    image_path=image_path,
                    output_path=os.path.join(output_dir, f"result_{i}"),
                )
                result["source_file"] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    "source_file": image_path,
                    "error": str(e)
                })
        
        # Save combined results
        combined_path = os.path.join(output_dir, "all_results.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Extract data from Turkish e-invoices")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--image", type=str,
                        help="Path to single invoice image")
    parser.add_argument("--image-dir", type=str,
                        help="Directory containing invoice images")
    parser.add_argument("--output-dir", type=str, default="./inference_output",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--base-size", type=int, default=1024,
                        help="Base image size")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Patch image size")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = InvoiceExtractor(
        model_path=args.model_path,
        device=args.device,
        base_size=args.base_size,
        image_size=args.image_size,
    )
    
    if args.image:
        # Single image extraction
        result = extractor.extract(
            image_path=args.image,
            output_path=args.output_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.image_dir:
        # Batch extraction
        import glob
        image_paths = glob.glob(os.path.join(args.image_dir, "*.jpg"))
        image_paths.extend(glob.glob(os.path.join(args.image_dir, "*.png")))
        
        results = extractor.batch_extract(
            image_paths=image_paths,
            output_dir=args.output_dir,
        )
        print(f"Processed {len(results)} images. Results saved to {args.output_dir}")
    
    else:
        print("Please provide --image or --image-dir argument")
        parser.print_help()


if __name__ == "__main__":
    main()
