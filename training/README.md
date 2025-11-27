# DeepSeek OCR Fine-tuning for Turkish E-Invoice Extraction

This training pipeline uses **Unsloth** to fine-tune DeepSeek OCR for extracting structured data from Turkish e-invoices (e-Fatura).

## 🚀 START HERE - Run Training

**For AI Assistant**: Read this section and execute the commands step by step.

### Step 1: Install Dependencies
```bash
cd /path/to/e-invoice-dataset
pip install --upgrade unsloth unsloth_zoo
pip install -r training/requirements.txt
```

### Step 2: Check GPU Availability
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step 3: Run Training with Unsloth (RECOMMENDED)
```bash
python training/train_unsloth.py
```

This uses Unsloth which is:
- **1.4x faster** than standard training
- **40% less VRAM** usage
- **5x longer context** support
- Uses `unsloth/DeepSeek-OCR` model (pre-fixed for latest transformers)

### Alternative: Run Full Pipeline
```bash
bash training/run_training.sh --all
```

---

## Why Unsloth?

| Feature | Unsloth | Standard |
|---------|---------|----------|
| **Speed** | 1.4x faster | Baseline |
| **VRAM** | 40% less | Full usage |
| **Context** | 5x longer | Limited |
| **Compatibility** | Latest transformers ✓ | May have issues |

## GPU Requirements

- **Minimum**: NVIDIA RTX 3090 (24GB VRAM) with LoRA
- **Recommended**: NVIDIA RTX 4090 (24GB) or A100 (40GB/80GB)
- **Cloud Options**: 
  - RunPod.io: RTX 4090 (~$0.44/hr) or A100 (~$1.89/hr)
  - Lambda Labs: A100 (~$1.10/hr)
  - Vast.ai: RTX 4090 (~$0.30/hr)

## Features

- **Automatic Dataset Preparation**: Converts JSON annotations and images to training format
- **LoRA Fine-tuning**: Parameter-efficient training with LoRA adapters
- **Multi-page Support**: Handles invoices spanning multiple pages
- **Turkish Language**: Optimized for Turkish e-invoice format
- **Structured Output**: Extracts data in a predefined JSON schema

## Quick Start (Single Command)

Run the complete training pipeline:

```bash
# Make sure you have the model weights first!
bash training/run_training.sh --all
```

## Project Structure

```
e-invoice-dataset/
├── images/                    # Invoice images (JPG/PNG)
├── output/                    # JSON annotations for each PDF
├── deepseek_ocr_code/         # DeepSeek OCR model code
├── training/
│   ├── prepare_dataset.py     # Dataset preparation script
│   ├── train.py               # Training script
│   ├── inference.py           # Inference script
│   ├── config.json            # Training configuration
│   ├── requirements.txt       # Python dependencies
│   ├── run_training.sh        # Main execution script
│   └── README.md              # This file
├── dataset/                   # Generated dataset (train/val/test splits)
├── checkpoints/               # Model checkpoints
└── logs/                      # Training logs
```

## Installation

1. Install Python dependencies:
```bash
pip install -r training/requirements.txt
```

2. Ensure CUDA is available for GPU training:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

### Step 1: Prepare Dataset

Convert your annotations and images to training format:

```bash
python training/prepare_dataset.py \
    --output-dir ./output \
    --images-dir ./images \
    --dataset-dir ./dataset
```

### Step 2: Configure Training

Edit `training/config.json` to customize training parameters:

```json
{
    "training": {
        "num_train_epochs": 10,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8
    },
    "model": {
        "use_lora": true,
        "lora_r": 64,
        "lora_alpha": 128
    }
}
```

### Step 3: Run Training

```bash
python training/train.py --config training/config.json
```

### Step 4: Run Inference

```bash
python training/inference.py \
    --model-path ./checkpoints \
    --image ./images/sample_invoice.jpg \
    --output-dir ./inference_output
```

## Output Schema

The model extracts the following fields from invoices:

```json
{
    "invoice_currency": "EUR",
    "invoice_customer_name": "Customer Company",
    "invoice_customer_address": "Full address",
    "invoice_customer_country": "Country",
    "invoice_date": "DD-MM-YYYY",
    "invoice_delivery_term": "DAP/FOB/EXW",
    "invoice_id": "Invoice number",
    "invoice_payment_term": "Payment terms",
    "invoice_po_number": "PO number",
    "invoice_shipment_country_of_origin": "Origin country",
    "invoice_supplier_name": "Supplier company",
    "invoice_supplier_address": "Supplier address",
    "invoice_supplier_country": "Supplier country",
    "invoice_supplier_vkn": "Tax ID",
    "invoice_total_amount": "Total amount",
    "invoice_total_package_quantity": "Package count",
    "invoice_total_quantity": "Total quantity",
    "invoice_total_gross_weight": "Gross weight",
    "invoice_total_net_weight": "Net weight",
    "items": [
        {
            "invoice_item_no": "1",
            "invoice_item_description": "Product description",
            "invoice_item_commodity_code": "HS code",
            "invoice_item_country_of_origin": "Origin",
            "invoice_item_product_id": "Product ID",
            "invoice_item_quantity": "Quantity",
            "invoice_item_unit_price": "Unit price",
            "invoice_item_total_amount": "Line total",
            "invoice_item_unit_type": "Unit type",
            "invoice_item_package_quantity": "Package qty"
        }
    ]
}
```

## Training Tips

1. **GPU Memory**: Reduce `per_device_train_batch_size` if you run out of GPU memory
2. **Learning Rate**: Start with `2e-5` for LoRA, `1e-5` for full fine-tuning
3. **Epochs**: 5-10 epochs is usually sufficient for small datasets
4. **Validation**: Monitor `eval_loss` to detect overfitting

## Hardware Requirements

- **Minimum**: 16GB GPU VRAM (with LoRA)
- **Recommended**: 24GB+ GPU VRAM (e.g., RTX 3090, RTX 4090, A100)
- **RAM**: 32GB system memory

## Troubleshooting

### Out of Memory
- Enable gradient checkpointing in config
- Reduce batch size
- Use LoRA instead of full fine-tuning

### Slow Training
- Enable bf16 mixed precision
- Increase number of dataloader workers
- Use SSD storage for dataset

## License

This project follows the license terms of the DeepSeek OCR model.

## Citation

If you use this training pipeline, please cite:

```bibtex
@misc{deepseek-ocr-invoice,
    title={DeepSeek OCR Fine-tuning for Turkish E-Invoice Extraction},
    author={Your Name},
    year={2025},
    publisher={GitHub},
}
```
