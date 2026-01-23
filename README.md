# NBA Predictions

NBA player prop predictions using `nba_api` and machine learning, with an interactive desktop GUI.

The app allows you to input player lines and betting odds, then generates model-based predictions using previous season data.

---

## Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

---

## Install (venv for isolated instal(recommened))

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install "git+https://github.com/bmart231/NBA_Predictions.git"
```

---

## macOS/Linux

```bash
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

---

## Running

```bash
python -m src.ui.app
```

---

## Running log

```bash
nba-ui
```

---

# Input Format

You'll be prompted to enter:

- **Player Name** (e.g., LeBron James)
- **Points Line** (e.g., 27.5)
- **Rebounds Line** (e.g., 7.5)
- **Over Odds** (e.g., -110)
- **Under Odds** (e.g., -110)

> **Note:** If player data is unavailable, a warning will be displayed.

> > **Note:** More data (AST, MIN, STL coming soon)
