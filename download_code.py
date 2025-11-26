from huggingface_hub import snapshot_download

snapshot_download(repo_id="deepseek-ai/DeepSeek-OCR", allow_patterns=["*.py", "*.json"], local_dir="deepseek_ocr_code")
