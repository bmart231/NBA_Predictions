# venv activation

.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

same thing on macos but js use venv.activate/...
run python -m pip install -r requirements.txt to install on dependencies needed to run all the files

# running client

Once venv (virtual environment) has been activated just run
python -m src.model.predict_props or (py in powershell)
to run

# Terminal Input

Should include:
Player Name:
Pts:
Reb:
Over:
Under:

_if data for player is not available will throwing warning_
