import json
import glob
import os
import hashlib

def convert_to_string(data, key=None):
    """
    Recursively convert all values in data to strings, except for 'page' which should be int.
    """
    if isinstance(data, dict):
        return {k: convert_to_string(v, k) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_string(v, key) for v in data]
    elif key == "page":
        try:
            return int(data)
        except (ValueError, TypeError):
            return data # Keep as is if not convertible, though it should be int
    elif data is None:
        return "" # Convert null to empty string
    else:
        return str(data)

def get_content_hash(data):
    """
    Generate a hash for the data content to identify duplicates.
    """
    # Sort keys to ensure consistent ordering for hashing
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()

def main():
    output_dir = "output"
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    print(f"Found {len(json_files)} files in {output_dir}")
    
    all_invoices = []
    seen_hashes = set()
    duplicates = []
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
            
            # Convert to string
            cleaned_content = convert_to_string(content)
            
            # Save back immediately
            with open(json_file, 'w') as f:
                json.dump(cleaned_content, f, indent=2, ensure_ascii=False)
            
            # Duplicate Check
            # Assuming content is a list of pages, and each page has "data" which is a list of invoices
            if isinstance(cleaned_content, list):
                for page in cleaned_content:
                    if isinstance(page, dict) and "data" in page and isinstance(page["data"], list):
                        for invoice in page["data"]:
                            # Hash the invoice object
                            inv_hash = get_content_hash(invoice)
                            
                            if inv_hash in seen_hashes:
                                duplicates.append((filename, page.get("page", "unknown"), invoice.get("invoice_id", "unknown")))
                            else:
                                seen_hashes.add(inv_hash)
                                all_invoices.append(invoice)
                                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("-" * 30)
    print("Processing Complete.")
    print(f"Total unique invoices found: {len(seen_hashes)}")
    
    if duplicates:
        print(f"Found {len(duplicates)} duplicates:")
        for fname, page, inv_id in duplicates:
            print(f"  - File: {fname}, Page: {page}, Invoice ID: {inv_id}")
    else:
        print("No duplicates found.")

if __name__ == "__main__":
    main()
