from __future__ import annotations
from pathlib import Path

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QComboBox,
    QPushButton, QTextEdit, QMessageBox
)


# Import the run_prediction function
from src.model.prop_engine import run_prediction

features_path = Path("data/processed/player_features_2023_2026.parquet") # path to processed features
model_path = Path("models") # location of trained models

class App(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NBA Player Prop Predictor")
        self.setMinimumWidth(800)
        
        self.player_input = QLineEdit("LeBron James") # Player name input
        
        # 
        self.pts_line = QDoubleSpinBox() # Points line input
        self.pts_line.setRange(0, 200) # set valid range
        self.pts_line.setSingleStep(0.5) # step size
        self.pts_line.setValue(27.5) # default value

        self.reb_line = QDoubleSpinBox() # Rebounds line input
        self.reb_line.setRange(0, 100) # set valid range
        self.reb_line.setSingleStep(0.5) # step size
        self.reb_line.setValue(7.5) # default value

        self.over_odds = QDoubleSpinBox() # Over odds input
        self.over_odds.setRange(-10000, 10000) # set valid range
        self.over_odds.setSingleStep(5)
        self.over_odds.setValue(-110)

        self.under_odds = QDoubleSpinBox() # Under odds input
        self.under_odds.setRange(-10000, 10000)
        self.under_odds.setSingleStep(5)
        self.under_odds.setValue(-110)

        self.sigma_window = QComboBox() # Sigma window selection
        self.sigma_window.addItems(["5", "10", "20"])
        self.sigma_window.setCurrentText("20")

        self.season_input = QLineEdit() # Season input
        self.season_input.setPlaceholderText('Optional: 2025-26')

        self.predict_btn = QPushButton("Predict") # Predict button
        self.predict_btn.clicked.connect(self.on_predict)

        self.output = QTextEdit() # Output display
        self.output.setReadOnly(True) # make it read-only

        layout = QVBoxLayout() # main layout

        layout.addWidget(QLabel("Player name:")) # player name label
        layout.addWidget(self.player_input)

        row1 = QHBoxLayout() # first row layout
        row1.addWidget(QLabel("PTS line:")) # points line label
        row1.addWidget(self.pts_line) # points line input
        row1.addWidget(QLabel("REB line:")) # rebounds line label
        row1.addWidget(self.reb_line) # rebounds line input
        layout.addLayout(row1) # add first row to main layout

        row2 = QHBoxLayout() # second row layout
        row2.addWidget(QLabel("Over odds:")) # over odds label
        row2.addWidget(self.over_odds) # over odds input
        row2.addWidget(QLabel("Under odds:")) # under odds label
        row2.addWidget(self.under_odds) # under odds input
        layout.addLayout(row2) # add second row to main layout

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Sigma window:"))
        row3.addWidget(self.sigma_window)
        row3.addWidget(QLabel("Season:"))
        row3.addWidget(self.season_input)
        row3.addWidget(self.predict_btn)
        layout.addLayout(row3)

        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output)

        self.setLayout(layout)

    def on_predict(self) -> None:
        try:
            player = self.player_input.text().strip()
            if not player: 
                raise ValueError("Player name cannot be empty.")
            
            # optional season input
            season = self.season_input.text().strip() or None
            
            # results from the prediction function
            res = run_prediction(
                features_path=features_path, # path to processed features
                model_path=model_path, # path to trained models
                    player_name=player,
                    pts_line=float(self.pts_line.value()),
                    reb_line=float(self.reb_line.value()),
                    sigma_window=int(self.sigma_window.currentText()),
                    over_odds=float(self.over_odds.value()),
                    under_odds=float(self.under_odds.value()),
                    season=season,
            )
            BOLD = '\033[1m'
            END = '\033[0m'
            lines = []
            lines.append(f"Player: {res.player_name}")
            lines.append(f"Latest game date: {res.game_date.date()}")
            if season:
                lines.append(f"Season filter: {season}")
            lines.append(f"Sigma window: {res.sigma_window}")
            lines.append("")
            lines.append("POINTS")
            lines.append(f"  mu: {res.mu_pts:.2f}   sigma: {res.sigma_pts:.2f}   line: {res.pts_line}")
            lines.append(f"  P(Over): {res.p_over_pts:.3f}   P(Under): {res.p_under_pts:.3f}")
            lines.append(f"  Fair odds Over: {res.fair_over_pts:+d}   Under: {res.fair_under_pts:+d}")
            lines.append(f"  EV Over: {res.ev_over_pts:+.3f} per $1   EV Under: {res.ev_under_pts:+.3f} per $1")
            lines.append("")
            lines.append("REBOUNDS")
            lines.append(f"  mu: {res.mu_reb:.2f}   sigma: {res.sigma_reb:.2f}   line: {res.reb_line}")
            lines.append(f"  P(Over): {res.p_over_reb:.3f}   P(Under): {res.p_under_reb:.3f}")
            lines.append(f"  Fair odds Over: {res.fair_over_reb:+d}   Under: {res.fair_under_reb:+d}")
            lines.append(f"  EV Over: {res.ev_over_reb:+.3f} per $1   EV Under: {res.ev_under_reb:+.3f} per $1")
            


            
            # Fair odds and EV for rebounds
            self.output.setPlainText("\n".join(lines))
            
        except Exception as e: # catch all exceptions
            QMessageBox.critical(self, "Error", str(e)) # show error message box   
            
        
        
        
def main() -> None:
    app = QApplication([])
    window = App()
    window.show()
    app.exec()
    
if __name__ == "__main__":
    main()      