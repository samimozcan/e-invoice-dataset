import os
import json
import glob
from pdf2image import convert_from_path

def create_dataset():
    # Define the schema
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
    ]
    
    query_schema = json.dumps(schema)
    
    base_query = "Extract invoice information from the images"
    
    # Ensure images directory exists
    os.makedirs("images", exist_ok=True)
    
    pdf_files = glob.glob("pdfs/*.pdf")
    dataset = []
    
    print(f"Found {len(pdf_files)} PDF files.")
    
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        json_filename = f"{filename}.json"
        json_path = os.path.join("output", json_filename)
        
        if not os.path.exists(json_path):
            print(f"Warning: JSON output not found for {filename}, skipping.")
            continue
            
        try:
            with open(json_path, 'r') as f:
                response_data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON for {filename}: {e}")
            continue
            
        print(f"Processing {filename}...")
        
        try:
            images = convert_from_path(pdf_file)
        except Exception as e:
            print(f"Error converting {filename} to images: {e}")
            continue
            
        image_paths = []
        for i, image in enumerate(images):
            page_num = i + 1
            image_filename = f"{filename}_page_{page_num}.jpg"
            image_path = os.path.join("images", image_filename)
            image.save(image_path, "JPEG")
            image_paths.append(image_path)
            
        # Create prompt with multiple <image> tokens
        # Format: <image>\n<image>\n... Query...
        image_tokens = "<image>\n" * len(image_paths)
        full_prompt = (image_tokens + base_query + ". return response in JSON format, by strictly following this JSON schema: " + query_schema +
                     ". If a field is not visible or cannot be found in the document, return null. Do not guess, infer, or generate values for missing fields.")
        
        # Create dataset entry
        entry = {
            "images": image_paths,
            "prompt": full_prompt,
            "response": json.dumps(response_data, ensure_ascii=False) # Response should be string for training
        }
        
        dataset.append(entry)
            
    # Save dataset
    with open("dataset_multi_image.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Dataset created with {len(dataset)} entries. Saved to dataset_multi_image.jsonl")

if __name__ == "__main__":
    create_dataset()
