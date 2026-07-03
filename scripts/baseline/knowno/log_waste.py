from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class RunLogger:
    def __init__(
        self,
        script_path: str,
        log_file: str = "",
        verbose: bool = False,
        prefix: str = "knowno_multistep",
        write_immediately: bool = True,
    ):
        log_dir = Path(script_path).resolve().parent / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(log_file) if log_file else log_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.file = self.path.open("w", encoding="utf-8")
        self.verbose = verbose
        self.write_enabled = write_immediately
        self.buffer = []

    def enable_file_logging(self):
        self.write_enabled = True

    def flush_buffer(self):
        for text in self.buffer:
            self.file.write(text + "\n")
        self.file.flush()
        self.buffer.clear()

    def discard_buffer_from_marker(self, marker: str):
        for index, text in enumerate(self.buffer):
            if marker in text:
                del self.buffer[index:]
                return

    def file_only(self, *values):
        text = " ".join(str(value) for value in values)
        if self.write_enabled:
            self.file.write(text + "\n")
            self.file.flush()
        else:
            self.buffer.append(text)
        if self.verbose:
            print(text)

    def console(self, *values):
        text = " ".join(str(value) for value in values)
        print(text)
        if self.write_enabled:
            self.file.write(text + "\n")
            self.file.flush()
        else:
            self.buffer.append(text)

    def colored(self, text: str, plain_text: str):
        print(text)
        if self.write_enabled:
            self.file.write(plain_text + "\n")
            self.file.flush()
        else:
            self.buffer.append(plain_text)

    def json(self, title: str, data):
        self.file_only(title)
        self.file_only(json.dumps(data, indent=2, sort_keys=True, default=str))

    def close(self):
        self.file.close()
