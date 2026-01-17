from huggingface_hub import hf_hub_download

#Download Ministral 3 8B Reasoning 8-bit Quantized GGUF model
model_path = hf_hub_download(
    repo_id="mistralai/Ministral-3-8B-Reasoning-2512-GGUF",
    filename="Ministral-3-8B-Reasoning-2512-Q8_0.gguf",
    cache_dir="./models")

print(f"Model downloaded to: {model_path}")