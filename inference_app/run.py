"""Start the hotel review web page.

Run:  python run.py
The page opens in your browser (usually http://localhost:8501). Stop with Ctrl+C.
"""

import subprocess
import sys
from pathlib import Path

app_file = Path(__file__).resolve().parent / "app.py"

# "python -m streamlit" works even if the streamlit command is not on your PATH.
subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file)])
