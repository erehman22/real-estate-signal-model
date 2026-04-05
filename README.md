# Real Estate Signal Model

Early-warning signal detection for property market booms.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Structure
- `config/` — location metadata and settings
- `pipeline/` — data collection scripts (run in order 01 → 06)
- `data/raw/` — raw collected data (not tracked in git)
- `data/processed/` — cleaned, feature-engineered data
- `analysis/` — Jupyter notebooks for validation
- `model/` — training and inference scripts
- `docs/` — research documents
