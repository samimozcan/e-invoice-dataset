import json
import glob
import os

def clean_output():
    json_files = glob.glob("output/*.json")
    print(f"Found {len(json_files)} JSON files to clean.")
    
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                content = json.load(f)
            
            # Sparrow output is usually a list of page results:
            # [{"data": [...], "page": 1}, ...]
            # Or sometimes just the data if it's single page and configured differently.
            # Our script uses default which returns the list of pages.
            
            cleaned_data = []
            
            if isinstance(content, list):
                for page in content:
                    if isinstance(page, dict) and "data" in page:
                        # "data" is a list of objects found on the page
                        if isinstance(page["data"], list):
                            cleaned_data.extend(page["data"])
                        else:
                            # Should be list, but just in case
                            cleaned_data.append(page["data"])
                    else:
                        # Maybe it's already the object?
                        cleaned_data.append(page)
            elif isinstance(content, dict):
                if "data" in content:
                     if isinstance(content["data"], list):
                        cleaned_data.extend(content["data"])
                     else:
                        cleaned_data.append(content["data"])
                else:
                    cleaned_data.append(content)
            
            # Write back the cleaned data
            with open(json_file, "w") as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
                
            print(f"Cleaned {json_file}")
            
        except Exception as e:
            print(f"Error cleaning {json_file}: {e}")

if __name__ == "__main__":
    clean_output()
