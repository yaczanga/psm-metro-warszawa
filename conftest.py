"""Udostepnia pakiet src testom uruchamianym z korzenia repozytorium."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
