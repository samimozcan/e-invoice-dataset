import sys
import os
import json
import glob
import time

# Add sparrow-ml/llm to sys.path
sys.path.append(os.path.abspath("sparrow_repo/sparrow-ml/llm"))

from pipelines.interface import get_pipeline

def process_invoices():
    # Define the schema based on ai.types.ts
    # Sparrow uses a JSON-like structure for schema definition
    schema = [
        {
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
    ]
    
    schema_str = json.dumps(schema)
    
    # Configuration
    pipeline_name = "sparrow-parse"
    # Using Qwen2.5-VL-7B for speed/memory, as it's a "Quick Testing" recommendation.
    # If accuracy is low, we can switch to 72B.
    model_name = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit" 
    options = ["mlx", model_name]
    
    # Initialize pipeline
    print(f"Initializing pipeline {pipeline_name} with model {model_name}...")
    try:
        rag = get_pipeline(pipeline_name)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Get all PDF files
    pdf_files = glob.glob("pdfs/*.pdf")
    print(f"Found {len(pdf_files)} PDF files.")
    
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        output_file = os.path.join("output", f"{filename}.json")
        
        if os.path.exists(output_file):
            print(f"Skipping {filename}, already processed.")
            continue
            
        print(f"Processing {filename}...")
        start_time = time.time()
        
        try:
            # Run pipeline
            # run_pipeline signature: 
            # run_pipeline(self, pipeline_name, query, file_path, options, crop_size, instruction, validation, precision, page_type, debug_dir, debug, use_db)
            answer = rag.run_pipeline(
                pipeline_name,
                schema_str,
                pdf_file,
                options,
                None,   # crop_size
                False,  # instruction
                False,  # validation
                False,  # precision
                None,   # page_type
                None,   # debug_dir
                False,  # debug
                False   # use_db
            )
            
            # Save result
            with open(output_file, "w") as f:
                # answer is usually a dict or list. 
                # If it's a string, try to parse it.
                if isinstance(answer, str):
                    try:
                        answer = json.loads(answer)
                    except:
                        pass
                
                json.dump(answer, f, indent=2, ensure_ascii=False)
                
            print(f"Finished {filename} in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # Continue to next file

if __name__ == "__main__":
    process_invoices()
