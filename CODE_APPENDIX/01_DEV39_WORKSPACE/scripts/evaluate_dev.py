
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework_src.evaluation.metrics import evaluate_predictions
part = Path(__file__).resolve().parents[1]
with open(part/'outputs'/'dev_predictions.json','r',encoding='utf-8') as f: preds = json.load(f)
with open(part/'data'/'dev_gold.json','r',encoding='utf-8') as f: gold = json.load(f)
res = evaluate_predictions(preds, gold)
with open(part/'results'/'dev_evaluation.json','w',encoding='utf-8') as f: json.dump(res,f,ensure_ascii=False,indent=2)
print('Dev evaluation written.')
