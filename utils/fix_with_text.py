import json
import glob
import os
import subprocess
import re

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using pdftotext command line tool.
    """
    try:
        # Run pdftotext -layout <pdf_path> -
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def find_value_in_text(text, patterns):
    """
    Try to find a value in text using a list of regex patterns.
    """
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def fix_json_with_text(json_file, pdf_file):
    """
    Fix missing fields in JSON using text extracted from PDF.
    """
    try:
        with open(json_file, 'r') as f:
            content = json.load(f)
        
        # Check if we need to fix anything
        needs_fix = False
        
        # Extract text only if needed
        text = None
        
        # Helper to get text lazily
        def get_text():
            nonlocal text
            if text is None:
                print(f"  Extracting text from {pdf_file}...")
                text = extract_text_from_pdf(pdf_file)
            return text

        # Patterns for common missing fields
        patterns = {
            "invoice_total_gross_weight": [
                r"Brüt\s*Kg\s*[:\s]*([\d,.]+)",
                r"Gross\s*Weight\s*[:\s]*([\d,.]+)",
                r"Total\s*Gross\s*Weight\s*[:\s]*([\d,.]+)",
                r"KAP\s*:\s*\d+\s*BRÜT\s*KG\s*([\d,.]+)", # Specific to one of the samples
                r"Gross\s*Wt\s*[:\s]*([\d,.]+)"
            ],
            "invoice_total_net_weight": [
                r"Net\s*Kg\s*[:\s]*([\d,.]+)",
                r"Net\s*Weight\s*[:\s]*([\d,.]+)",
                r"Total\s*Net\s*Weight\s*[:\s]*([\d,.]+)",
                r"NET\s*KG\s*([\d,.]+)" # Specific to one of the samples
            ]
        }

        # Traverse and fix
        if isinstance(content, list):
            for page in content:
                if isinstance(page, dict) and "data" in page and isinstance(page["data"], list):
                    for invoice in page["data"]:
                        for key, value in invoice.items():
                            # Check for "null" string or empty string or None
                            if value == "null" or value == "" or value is None:
                                if key in patterns:
                                    current_text = get_text()
                                    found_value = find_value_in_text(current_text, patterns[key])
                                    
                                    if found_value:
                                        print(f"  Found {key}: {found_value}")
                                        invoice[key] = found_value
                                        needs_fix = True
        
        if needs_fix:
            print(f"  Updating {json_file}...")
            with open(json_file, 'w') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            print(f"  No fixes needed for {json_file}")

    except Exception as e:
        print(f"Error processing {json_file}: {e}")

def main():
    output_dir = "output"
    pdf_dir = "pdfs"
    
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    print(f"Found {len(json_files)} JSON files.")
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        # Remove .json extension to get pdf filename
        pdf_filename = filename[:-5] 
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        
        if os.path.exists(pdf_path):
            fix_json_with_text(json_file, pdf_path)
        else:
            print(f"PDF not found for {filename}: {pdf_path}")

if __name__ == "__main__":
    main()
