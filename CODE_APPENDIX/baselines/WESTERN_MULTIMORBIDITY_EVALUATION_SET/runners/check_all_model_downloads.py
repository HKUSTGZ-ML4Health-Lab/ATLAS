from pathlib import Path
import os

model_root = Path(os.environ.get("ATLAS_MODEL_ROOT", str(Path.home() / ".cache" / "atlas_models"))).expanduser()
models = {
    "Mistral-Small-3.2-24B-Instruct-2506": model_root / "Mistral-Small-3.2-24B-Instruct-2506",
    "MedGemma 27B Text": model_root / "medgemma-27b-text-it",
    "BGE-M3": model_root / "bge-m3",
    "Llama-3.3-70B-Instruct": model_root / "Llama-3.3-70B-Instruct",
}

for name, path in models.items():
    print("\n===", name, "===")
    print("path:", path)
    print("exists:", path.exists())
    cfg = path / "config.json"
    print("config:", cfg.exists())
    if path.exists():
        files = list(path.glob("*"))
        safetensors = list(path.glob("*.safetensors"))
        print("num_files:", len(files))
        print("num_safetensors:", len(safetensors))
        total = sum(f.stat().st_size for f in files if f.is_file())
        print("size_GB:", round(total / 1024**3, 2))
