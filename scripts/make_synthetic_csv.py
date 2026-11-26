"""Write the synthetic data as a CSV in the canonical schema.

    python scripts/make_synthetic_csv.py                 ->  data/synthetic_demo.csv

Use the output as a template: real data must have exactly these columns (see data/README.md).
The file is SYNTHETIC. Do not name it, describe it or present it as market data.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iex_forecast.data import make_synthetic, to_long_frame  # noqa: E402

out = ROOT / "data" / "synthetic_demo.csv"
frame = to_long_frame(make_synthetic())
frame.to_csv(out, index=False)
print("wrote", out, frame.shape)
