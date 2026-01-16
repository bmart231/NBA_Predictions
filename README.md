# Setup

## Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## macOS/Linux

```bash
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

# Running the Model

```bash
python -m src.model.predict_props
```

# Input Format

You'll be prompted to enter:

- **Player Name** (e.g., LeBron James)
- **Points Line** (e.g., 27.5)
- **Rebounds Line** (e.g., 7.5)
- **Over Odds** (e.g., -110)
- **Under Odds** (e.g., -110)

> **Note:** If player data is unavailable, a warning will be displayed.
