from modelscope import snapshot_download
from pathlib import Path
import os

model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
cache_dir = Path(os.environ.get("MODELSCOPE_CACHE", str(Path.home() / ".cache" / "modelscope"))).expanduser()
log_dir = Path(os.environ.get("ATLAS_MODEL_LOG_DIR", str(Path(__file__).resolve().parents[3] / "baselines_v2" / "model_logs"))).expanduser()
log_dir.mkdir(parents=True, exist_ok=True)

print("[INFO] start ModelScope download:", model_id, flush=True)
path = snapshot_download(model_id=model_id, cache_dir=str(cache_dir))
print("[OK] downloaded to:", path, flush=True)
(log_dir / 'deepseek_r1_distill_qwen32b_modelscope_path.txt').write_text(str(path), encoding="utf-8")
