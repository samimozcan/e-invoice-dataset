import os
import glob
import json

def fix_json_encoding():
    output_dir = "output"
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Write back with ensure_ascii=False to write actual characters instead of \uXXXX
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"Fixed encoding for {os.path.basename(json_file)}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    fix_json_encoding()
