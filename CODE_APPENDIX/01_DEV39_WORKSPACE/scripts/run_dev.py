
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework_src.agents.unified_orchestrator import UnifiedOrchestratorAgent
part = Path(__file__).resolve().parents[1]
with open(part/'data'/'dev_input.json','r',encoding='utf-8') as f: cases = json.load(f)
with open(part/'rules'/'working_rules.json','r',encoding='utf-8') as f: rules = json.load(f)
preds = [UnifiedOrchestratorAgent(rules).run_case(c) for c in cases]
with open(part/'outputs'/'dev_predictions.json','w',encoding='utf-8') as f: json.dump(preds,f,ensure_ascii=False,indent=2)
print(f'Wrote {len(preds)} dev predictions.')
