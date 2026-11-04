"""Render output/results.json into one self-contained HTML page."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "output/results.json").read_text())
tpl = (ROOT / "stockout/template.html").read_text()
(ROOT / "output/index.html").write_text(tpl.replace("/*DATA*/null", json.dumps(r, separators=(",", ":")).replace("</", "<\\/")))
print("wrote output/index.html")
