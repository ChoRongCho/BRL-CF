from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class RunLogger:
    def __init__(self, script_path: str, log_file: str = "", verbose: bool = False, prefix: str = "knowno_multistep"):
        log_dir = Path(script_path).resolve().parent / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(log_file) if log_file else log_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.file = self.path.open("w", encoding="utf-8")
        self.verbose = verbose

    def file_only(self, *values):
        text = " ".join(str(value) for value in values)
        self.file.write(text + "\n")
        self.file.flush()
        if self.verbose:
            print(text)

    def console(self, *values):
        text = " ".join(str(value) for value in values)
        print(text)
        self.file.write(text + "\n")
        self.file.flush()

    def colored(self, text: str, plain_text: str):
        print(text)
        self.file.write(plain_text + "\n")
        self.file.flush()

    def json(self, title: str, data):
        self.file_only(title)
        self.file_only(json.dumps(data, indent=2, sort_keys=True, default=str))

    def close(self):
        self.file.close()
