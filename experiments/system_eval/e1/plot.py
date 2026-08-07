#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _pipeline import plot_main
if __name__ == "__main__":
    plot_main("e1", Path(__file__).resolve().parent)
