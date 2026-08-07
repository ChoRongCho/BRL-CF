#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _pipeline import analysis_main
if __name__ == "__main__":
    analysis_main("e2", Path(__file__).resolve().parent)
