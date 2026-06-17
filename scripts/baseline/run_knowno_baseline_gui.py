from __future__ import annotations

import os
import queue
import random
import re
import ast
import json
import signal
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXPERIMENT_SCRIPT = SCRIPT_DIR / "knowno_baseline_experiment.py"
SETTINGS_PATH = SCRIPT_DIR / "llm_setting.json"


DEFAULTS = {
    "tomato": {
        "qhat": "0.92",
        "detect_success_prob": "0.85",
        "detect_label_error_prob": "0.05",
        "scan_success_prob": "0.9",
        "scan_label_error_prob": "0.15",
        "navigate_failure_prob": "0.05",
        "pick_failure_prob": "0.05",
        "place_failure_prob": "0.01",
        "discard_failure_prob": "0.01",
    },
    "wastesorting": {
        "qhat": "0.92",
        "detect_success_prob": "0.9",
        "detect_label_error_prob": "0.2",
    },
}


class KnownoGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("KnowNo Baseline Runner")
        self.geometry("1180x760")
        self.minsize(980, 620)
        self._configure_fonts()

        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str | tuple[str, int]] = queue.Queue()
        self.capture_mode = ""
        self.current_state_lines: list[str] = []
        self.current_option_lines: list[str] = []
        self.current_options: dict[str, str] = {}
        self.current_prediction_set: list[str] = []

        self.vars = {
            "domain": tk.StringVar(value="tomato"),
            "scene": tk.StringVar(value="01"),
            "prompt_version": tk.StringVar(value="v1"),
            "max_steps": tk.StringVar(value="50"),
            "seed": tk.StringVar(value=""),
            "score_temperature": tk.StringVar(value="3.0"),
            "qhat": tk.StringVar(value=DEFAULTS["tomato"]["qhat"]),
            "detect_success_prob": tk.StringVar(value=DEFAULTS["tomato"]["detect_success_prob"]),
            "detect_label_error_prob": tk.StringVar(value=DEFAULTS["tomato"]["detect_label_error_prob"]),
            "scan_success_prob": tk.StringVar(value=DEFAULTS["tomato"]["scan_success_prob"]),
            "scan_label_error_prob": tk.StringVar(value=DEFAULTS["tomato"]["scan_label_error_prob"]),
            "navigate_failure_prob": tk.StringVar(value=DEFAULTS["tomato"]["navigate_failure_prob"]),
            "pick_failure_prob": tk.StringVar(value=DEFAULTS["tomato"]["pick_failure_prob"]),
            "place_failure_prob": tk.StringVar(value=DEFAULTS["tomato"]["place_failure_prob"]),
            "discard_failure_prob": tk.StringVar(value=DEFAULTS["tomato"]["discard_failure_prob"]),
            "log_file": tk.StringVar(value=""),
            "verbose": tk.BooleanVar(value=False),
            "dry_run": tk.BooleanVar(value=False),
        }

        self._build_ui()
        self._apply_domain_defaults()
        self.after(100, self._drain_output_queue)

    def _configure_fonts(self) -> None:
        self.ui_font_family = self._pick_font_family(
            ["Noto Sans", "Ubuntu", "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"]
        )
        self.text_font_family = self._pick_font_family(
            ["Noto Sans", "Ubuntu", "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"]
        )
        default_font = tkfont.nametofont("TkDefaultFont")
        text_font = tkfont.nametofont("TkTextFont")
        fixed_font = tkfont.nametofont("TkFixedFont")
        default_font.configure(family=self.ui_font_family, size=13)
        text_font.configure(family=self.ui_font_family, size=13)
        fixed_font.configure(family=self.text_font_family, size=14)
        self.option_add("*Font", default_font)
        style = ttk.Style(self)
        style.configure(".", font=(self.ui_font_family, 13))
        style.configure("TButton", font=(self.ui_font_family, 13))
        style.configure("TLabel", font=(self.ui_font_family, 13))
        style.configure("TCheckbutton", font=(self.ui_font_family, 13))
        style.configure("TCombobox", font=(self.ui_font_family, 13))

    def _pick_font_family(self, candidates: list[str]) -> str:
        available = set(tkfont.families(self))
        for candidate in candidates:
            if candidate in available:
                return candidate
        return candidates[-1]

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=12)
        controls.grid(row=0, column=0, sticky="ns")
        output = ttk.Frame(self, padding=(0, 12, 12, 12))
        output.grid(row=0, column=1, sticky="nsew")
        output.rowconfigure(1, weight=1)
        output.columnconfigure(0, weight=1)

        self._build_controls(controls)
        self._build_output(output)

    def _build_controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(parent, text="Run Settings", font=("", 15, "bold")).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        row = self._combo_row(parent, row, "Domain", "domain", ["tomato", "wastesorting"])
        row = self._combo_row(parent, row, "Scene", "scene", ["01", "02", "03", "04", "05"])
        row = self._combo_row(parent, row, "Prompt", "prompt_version", ["v1", "v2"])
        row = self._entry_row(parent, row, "Max steps", "max_steps")
        row = self._entry_row(parent, row, "Seed", "seed")
        ttk.Label(parent, text="blank = random each run").grid(row=row - 1, column=2, sticky="w", padx=(6, 0))
        row = self._entry_row(parent, row, "Temperature", "score_temperature")
        row = self._entry_row(parent, row, "qhat", "qhat")

        ttk.Separator(parent).grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1
        ttk.Label(parent, text="Probabilities", font=("", 15, "bold")).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        row = self._entry_row(parent, row, "Detect success", "detect_success_prob")
        row = self._entry_row(parent, row, "Detect error", "detect_label_error_prob")
        row = self._entry_row(parent, row, "Scan success", "scan_success_prob")
        row = self._entry_row(parent, row, "Scan error", "scan_label_error_prob")
        row = self._entry_row(parent, row, "Navigate fail", "navigate_failure_prob")
        row = self._entry_row(parent, row, "Pick fail", "pick_failure_prob")
        row = self._entry_row(parent, row, "Place fail", "place_failure_prob")
        row = self._entry_row(parent, row, "Discard fail", "discard_failure_prob")

        ttk.Separator(parent).grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1
        row = self._entry_row(parent, row, "Log file", "log_file")
        ttk.Button(parent, text="Browse", command=self._choose_log_file).grid(row=row - 1, column=2, padx=(6, 0), sticky="ew")

        ttk.Checkbutton(parent, text="Verbose", variable=self.vars["verbose"]).grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(parent, text="Dry run", variable=self.vars["dry_run"]).grid(row=row, column=1, sticky="w", pady=(6, 0))
        row += 1

        button_row = ttk.Frame(parent)
        button_row.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(button_row, text="Start", command=self._start_run)
        self.stop_button = ttk.Button(button_row, text="Stop", command=self._stop_run, state="disabled")
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        row += 1

        ttk.Button(parent, text="Clear Display", command=self._clear_output).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        row += 1

        truth_frame = ttk.Frame(parent)
        truth_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        parent.rowconfigure(row, weight=1)
        truth_frame.rowconfigure(1, weight=1)
        truth_frame.columnconfigure(0, weight=1)
        ttk.Label(truth_frame, text="Ground Truth", font=(self.ui_font_family, 15, "bold")).grid(row=0, column=0, sticky="w")
        self.truth_text = tk.Text(truth_frame, wrap="word", height=14, state="disabled", font=(self.text_font_family, 13))
        truth_scrollbar = ttk.Scrollbar(truth_frame, orient="vertical", command=self.truth_text.yview)
        self.truth_text.configure(yscrollcommand=truth_scrollbar.set)
        self.truth_text.grid(row=1, column=0, sticky="nsew")
        truth_scrollbar.grid(row=1, column=1, sticky="ns")
        self._update_ground_truth()

        self.vars["domain"].trace_add("write", lambda *_: self._apply_domain_defaults())
        self.vars["scene"].trace_add("write", lambda *_: self._update_ground_truth())

    def _build_output(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(top, text="Run View", font=("", 15, "bold")).grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=1, sticky="e")

        panes = ttk.Panedwindow(parent, orient="vertical")
        panes.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        state_frame, self.state_text = self._text_panel(panes, "Current State", height=14)
        options_frame, self.options_text = self._text_panel(panes, "Options", height=10)
        events_frame, self.events_text = self._text_panel(panes, "Events", height=7)
        panes.add(state_frame, weight=3)
        panes.add(options_frame, weight=2)
        panes.add(events_frame, weight=1)

        input_frame = ttk.Frame(parent)
        input_frame.grid(row=2, column=0, sticky="ew")
        input_frame.columnconfigure(6, weight=1)
        ttk.Label(input_frame, text="Help answer").grid(row=0, column=0, sticky="w")
        for idx, token in enumerate(["A", "B", "C", "D", "E"], start=1):
            ttk.Button(input_frame, text=token, width=4, command=lambda t=token: self._send_choice(t)).grid(
                row=0, column=idx, padx=3
            )
        self.manual_input = tk.StringVar(value="")
        manual = ttk.Entry(input_frame, textvariable=self.manual_input)
        manual.grid(row=0, column=6, sticky="ew", padx=(10, 4))
        manual.bind("<Return>", lambda _event: self._send_manual_input())
        ttk.Button(input_frame, text="Send", command=self._send_manual_input).grid(row=0, column=7)

    def _text_panel(self, parent: ttk.Panedwindow, title: str, height: int) -> tuple[ttk.Frame, tk.Text]:
        frame = ttk.Frame(parent)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=title, font=("", 14, "bold")).grid(row=0, column=0, sticky="w")
        text = tk.Text(frame, wrap="word", height=height, state="disabled", font=(self.text_font_family, 14))
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        return frame, text

    def _combo_row(self, parent: ttk.Frame, row: int, label: str, key: str, values: list[str]) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        combo = ttk.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly", width=18)
        combo.grid(row=row, column=1, sticky="ew", pady=3)
        return row + 1

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, key: str) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.vars[key], width=20).grid(row=row, column=1, sticky="ew", pady=3)
        return row + 1

    def _apply_domain_defaults(self) -> None:
        domain = self.vars["domain"].get()
        defaults = DEFAULTS[domain]
        for key, value in defaults.items():
            self.vars[key].set(value)
        tomato_state = "normal" if domain == "tomato" else "disabled"
        for key in [
            "scan_success_prob",
            "scan_label_error_prob",
            "navigate_failure_prob",
            "pick_failure_prob",
            "place_failure_prob",
            "discard_failure_prob",
        ]:
            # ttk does not expose a direct lookup by variable, so leave entries editable.
            # Domain-specific command construction ignores tomato-only values for waste.
            _ = tomato_state
        self._update_ground_truth()

    def _update_ground_truth(self) -> None:
        if not hasattr(self, "truth_text"):
            return
        domain = self.vars["domain"].get()
        scene = self.vars["scene"].get().strip()
        truth = self._ground_truth_text(domain, scene)
        self._set_text(self.truth_text, truth)

    def _ground_truth_text(self, domain: str, scene: str) -> str:
        normalized_domain = "wastesorting" if domain == "wastesorting" else "tomato"
        try:
            scene_number = int(scene)
        except ValueError:
            return f"Invalid scene: {scene}"
        scene_path = PROJECT_ROOT / "scripts" / "domain" / normalized_domain / f"scene_{scene_number:02d}.yaml"
        if not scene_path.exists():
            return f"Scene file not found:\n{scene_path}"

        facts = self._read_true_init_facts(scene_path)
        if not facts:
            return "No true_init facts found."
        if normalized_domain == "tomato":
            labels: list[str] = []
            locations: list[str] = []
            for fact in facts:
                prop_match = re.fullmatch(r"(ripe|unripe|rotten)\(([^)]+)\)", fact)
                loc_match = re.fullmatch(r"at\(([^,]+),([^)]+)\)", fact)
                if prop_match:
                    labels.append(f"{prop_match.group(2)}: {prop_match.group(1)}")
                elif loc_match:
                    locations.append(f"{loc_match.group(1)}: {loc_match.group(2)}")
            lines = ["Properties:"]
            lines.extend(f"  {line}" for line in sorted(labels, key=self._natural_text_key))
            lines.append("Locations:")
            lines.extend(f"  {line}" for line in sorted(locations, key=self._natural_text_key))
            return "\n".join(lines)

        labels = []
        for fact in facts:
            match = re.fullmatch(r"(general|plastic|paper|can)\(([^)]+)\)", fact)
            if match:
                labels.append(f"{match.group(2)}: {match.group(1)}")
        return "Waste labels:\n" + "\n".join(f"  {line}" for line in sorted(labels, key=self._natural_text_key))

    @staticmethod
    def _read_true_init_facts(scene_path: Path) -> list[str]:
        facts: list[str] = []
        in_true_init = False
        for raw_line in scene_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if re.fullmatch(r"true_init\s*:", line):
                in_true_init = True
                continue
            if in_true_init and re.fullmatch(r"[A-Za-z_][\w-]*\s*:", line):
                break
            if not in_true_init:
                continue
            match = re.match(r'-\s*["\']?([^"\']+)["\']?\s*$', line)
            if match:
                facts.append(re.sub(r"\s+", "", match.group(1)))
        return facts

    @staticmethod
    def _natural_text_key(text: str) -> tuple[str, int, str]:
        match = re.search(r"(\d+)", text)
        if not match:
            return text, 999999, text
        return text[: match.start()], int(match.group(1)), text[match.end() :]

    @staticmethod
    def _model_slug() -> str:
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "unknown"
        model = str(settings.get("model", "")).lower()
        if "gpt-3.5" in model or "gpt-35" in model:
            return "gpt35turbo"
        if "gpt-4" in model:
            return "gpt4"
        return re.sub(r"[^a-z0-9]+", "", model) or "unknown"

    def _choose_log_file(self) -> None:
        domain = "wastesorting" if self.vars["domain"].get() == "waste" else self.vars["domain"].get()
        try:
            scene = f"{int(self.vars['scene'].get()):02d}"
        except ValueError:
            scene = self.vars["scene"].get().strip() or "01"
        initialdir = (
            PROJECT_ROOT
            / "experiments_logs"
            / "system_log"
            / domain
            / f"scene_{scene}_step50"
            / f"when_knowno_{self._model_slug()}"
        )
        initialdir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Choose log file",
            initialdir=str(initialdir),
            defaultextension=".txt",
            filetypes=[("Text logs", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.vars["log_file"].set(path)

    def _build_command(self) -> list[str]:
        seed = self.vars["seed"].get().strip()
        if not seed:
            seed = str(random.randrange(0, 2**32 - 1))
            self.vars["seed"].set(seed)

        cmd = [
            sys.executable,
            str(EXPERIMENT_SCRIPT),
            "--domain",
            self.vars["domain"].get(),
            "--scene",
            self.vars["scene"].get(),
            "--prompt-version",
            self.vars["prompt_version"].get(),
            "--qhat",
            self.vars["qhat"].get(),
            "--temperature",
            self.vars["score_temperature"].get(),
            "--max-steps",
            self.vars["max_steps"].get(),
            "--seed",
            seed,
            "--detect-success-prob",
            self.vars["detect_success_prob"].get(),
            "--detect-label-error-prob",
            self.vars["detect_label_error_prob"].get(),
        ]

        if self.vars["domain"].get() == "tomato":
            cmd.extend(
                [
                    "--scan-success-prob",
                    self.vars["scan_success_prob"].get(),
                    "--scan-label-error-prob",
                    self.vars["scan_label_error_prob"].get(),
                    "--navigate-failure-prob",
                    self.vars["navigate_failure_prob"].get(),
                    "--pick-failure-prob",
                    self.vars["pick_failure_prob"].get(),
                    "--place-failure-prob",
                    self.vars["place_failure_prob"].get(),
                    "--discard-failure-prob",
                    self.vars["discard_failure_prob"].get(),
                ]
            )

        if self.vars["verbose"].get():
            cmd.append("--verbose")
        if self.vars["dry_run"].get():
            cmd.append("--dry-run")
        log_file = self.vars["log_file"].get().strip()
        if log_file:
            cmd.extend(["--log-file", log_file])
        return cmd

    def _start_run(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("Run active", "A run is already active.")
            return

        cmd = self._build_command()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._reset_run_display()
        self._append_events("$ " + " ".join(self._quote(part) for part in cmd) + "\n\n")
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            messagebox.showerror("Start failed", str(exc))
            self.process = None
            return

        self.status_var.set("Running")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self._reader_thread, daemon=True).start()
        threading.Thread(target=self._wait_thread, daemon=True).start()

    def _reader_thread(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for chunk in iter(self.process.stdout.readline, ""):
            self.output_queue.put(chunk)
        self.process.stdout.close()

    def _wait_thread(self) -> None:
        assert self.process is not None
        code = self.process.wait()
        self.output_queue.put(("done", code))

    def _stop_run(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except OSError:
            self.process.terminate()
        self.status_var.set("Stopping")

    def _send_choice(self, token: str) -> None:
        self._send_stdin(token + "\n")

    def _send_manual_input(self) -> None:
        text = self.manual_input.get().strip()
        if not text:
            return
        self.manual_input.set("")
        self._send_stdin(text + "\n")

    def _send_stdin(self, text: str) -> None:
        if self.process is None or self.process.poll() is not None or self.process.stdin is None:
            messagebox.showinfo("No active run", "Start a run before sending input.")
            return
        try:
            self.process.stdin.write(text)
            self.process.stdin.flush()
            self._append_events(f">>> {text}")
        except OSError as exc:
            messagebox.showerror("Input failed", str(exc))

    def _drain_output_queue(self) -> None:
        while True:
            try:
                item = self.output_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple) and item[0] == "done":
                code = item[1]
                self.status_var.set(f"Finished ({code})")
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self._append_events(f"\n[process exited with code {code}]\n")
            else:
                self._process_output(str(item))
        self.after(100, self._drain_output_queue)

    def _process_output(self, text: str) -> None:
        for line in text.splitlines():
            self._process_output_line(line.rstrip())

    def _process_output_line(self, line: str) -> None:
        line = self._clean_output_line(line)
        if not line:
            return

        if line.startswith("====== Step "):
            self.current_state_lines = [line]
            self.current_option_lines = []
            self.capture_mode = "state"
            self._set_state("\n".join(self.current_state_lines))
            self._set_options("")
            return

        if line == "Generated options:":
            self.capture_mode = "options"
            self.current_option_lines = []
            self.current_options = {}
            self.current_prediction_set = []
            self._render_options()
            return

        if line == "Option scores:":
            self.capture_mode = ""
            return

        if line.startswith("Prediction set:"):
            self.current_prediction_set = self._parse_prediction_set(line)
            self._render_options()
            return

        if line.startswith("Help needed"):
            self._append_events(line + "\n")
            return

        if line.startswith("Selected/Executed"):
            self._append_events(line + "\n")
            return

        if line.startswith("Planner selected") or line.startswith("All target") or line.startswith("Reached max steps"):
            self._append_events(line + "\n")
            return

        if line.startswith("====== Summary ======") or line.startswith("Success:") or line.startswith("Stop reason:"):
            self._append_events(line + "\n")
            return

        if self.capture_mode == "options":
            if self._is_option_line(line):
                token = line[0]
                self.current_options[token] = line
                self._render_options()
            elif line.startswith("Prediction set:"):
                self.current_prediction_set = self._parse_prediction_set(line)
                self._render_options()
            return

        if self._is_state_line(line):
            self.current_state_lines.append(line)
            self.capture_mode = "state"
            self._set_state("\n".join(self.current_state_lines))
            return

        if self.capture_mode == "state" and self._is_state_detail_line(line):
            self.current_state_lines.append(line)
            self._set_state("\n".join(self.current_state_lines))
            return

        if line.startswith(("Running:", "Instruction:", "Prompt version:", "qhat:", "Detailed log file:")):
            self._append_events(line + "\n")

    @staticmethod
    def _is_option_line(line: str) -> bool:
        return re.match(r"^[A-E]\)\s+", line) is not None

    @staticmethod
    def _clean_output_line(line: str) -> str:
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        return line.strip()

    @staticmethod
    def _is_state_line(line: str) -> bool:
        prefixes = (
            "Robot location:",
            "Active tomatoes:",
            "Tomato states:",
            "Held tomato:",
            "Loaded tomatoes:",
            "Discarded tomatoes:",
            "Remaining objects:",
            "Observed waste attributes:",
            "Held object:",
            "Object currently held",
            "Initial objects:",
            "Available bins:",
        )
        return line.startswith(prefixes)

    @staticmethod
    def _is_state_detail_line(line: str) -> bool:
        if line.startswith("True "):
            return False
        if line.startswith(("Generated options:", "Option scores:", "Prediction set:")):
            return False
        if ": " not in line:
            return False
        key = line.split(":", 1)[0].strip()
        return key.startswith(("tomato", "waste"))

    def _reset_run_display(self) -> None:
        self.capture_mode = ""
        self.current_state_lines = []
        self.current_option_lines = []
        self.current_options = {}
        self.current_prediction_set = []
        self._set_state("")
        self._set_options("")
        self._set_events("")

    def _render_options(self) -> None:
        lines: list[str] = []
        if self.current_prediction_set:
            lines.append("Prediction set options:")
            for token in self.current_prediction_set:
                option = self.current_options.get(token)
                if option:
                    lines.append(option)
                else:
                    lines.append(f"{token}) [option text not captured]")
            other_tokens = [
                token for token in ["A", "B", "C", "D", "E"]
                if token in self.current_options and token not in self.current_prediction_set
            ]
            if other_tokens:
                lines.append("")
                lines.append("Other options:")
                lines.extend(self.current_options[token] for token in other_tokens)
            lines.append("")
            lines.append("Prediction set: " + ", ".join(self.current_prediction_set))
        else:
            lines.append("Generated options:")
            for token in ["A", "B", "C", "D", "E"]:
                if token in self.current_options:
                    lines.append(self.current_options[token])
        self._set_options("\n".join(lines))

    @staticmethod
    def _parse_prediction_set(line: str) -> list[str]:
        _, _, value = line.partition(":")
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return re.findall(r"[A-E]", text)
        if isinstance(parsed, list):
            return [str(item).strip().upper() for item in parsed if str(item).strip().upper() in {"A", "B", "C", "D", "E"}]
        return []

    def _set_state(self, text: str) -> None:
        self._set_text(self.state_text, text)

    def _set_options(self, text: str) -> None:
        self._set_text(self.options_text, text)

    def _set_events(self, text: str) -> None:
        self._set_text(self.events_text, text)

    def _append_events(self, text: str) -> None:
        self._append_text(self.events_text, text)

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        if text:
            widget.insert("end", text)
            if not text.endswith("\n"):
                widget.insert("end", "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def _append_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _clear_output(self) -> None:
        self._reset_run_display()

    @staticmethod
    def _quote(value: str) -> str:
        if not value or any(char.isspace() for char in value):
            return "'" + value.replace("'", "'\"'\"'") + "'"
        return value


def main() -> None:
    app = KnownoGui()
    app.mainloop()


if __name__ == "__main__":
    main()
