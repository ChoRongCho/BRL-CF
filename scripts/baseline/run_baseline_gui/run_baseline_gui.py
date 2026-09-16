"""Launch one baseline at a time for debugging; never starts batch experiments."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import tkinter.font as tkfont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = PROJECT_ROOT / "scripts" / "baseline"
BASELINES = ("knowno", "targeted_query_pomdp", "introplan")
DOMAINS = ("tomato", "wastesorting")



def scenes(domain):
    if domain not in DOMAINS:
        return []
    return sorted(p.stem[len("scene_"):] for p in
                  (PROJECT_ROOT / "scripts" / "domain" / domain).glob("scene_*.yaml"))


def build_launch(domain, scene, baseline, seed, max_steps):
    """Use existing runners so their domain-specific defaults remain authoritative."""
    if domain not in DOMAINS or scene not in scenes(domain):
        raise ValueError("유효한 domain과 scene을 선택하세요.")
    if baseline not in BASELINES:
        raise ValueError("유효한 baseline을 선택하세요.")
    steps = int(max_steps)
    if steps < 1:
        raise ValueError("Max steps는 1 이상이어야 합니다.")
    seed_value = secrets.randbits(32) if not str(seed).strip() else int(seed)
    if not 0 <= seed_value < 2**32:
        raise ValueError("Seed는 0 이상 2^32 미만이어야 합니다.")
    overrides = {
        "DOMAIN": domain, "SCENE": scene, "SEED": str(seed_value),
        "MAX_STEPS": str(steps), "MAX_STEP": str(steps),
        "PYTHONUNBUFFERED": "1", "AUTO_ANSWER": "true",
        "VERBOSE": "true", "DRY_RUN": "false",
    }
    runner = (PROJECT_ROOT / "run" / "run_knowno_baseline.sh" if baseline == "knowno"
              else BASELINE_ROOT / "targeted_query_pomdp" / "run.sh")
    if baseline == "introplan":
        runner = BASELINE_ROOT / "introplan" / "run.sh"
    return ["bash", str(runner)], overrides


class BaselineGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BRL Baseline Debug Runner")
        self.geometry("1200x820")
        self.minsize(1000, 680)
        self.configure_fonts()
        self.process = None
        self.events = queue.Queue()
        self.closing = False
        self.domain = tk.StringVar(value="tomato")
        self.scene = tk.StringVar(value="01")
        self.baseline = tk.StringVar(value="knowno")
        self.seed = tk.StringVar(value="42")
        self.max_steps = tk.StringVar(value="50")
        self.status = tk.StringVar(value="대기")
        self.note = tk.StringVar()

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="baseline을 하나씩 실행하는 디버깅용 폴더").pack(anchor="w")
        choices = ttk.Frame(frame)
        choices.pack(fill="x", pady=12)
        self.selectors = []
        for col, (label, variable, values) in enumerate([
            ("1. Domain", self.domain, DOMAINS),
            ("2. Scene", self.scene, scenes("tomato")),
            ("3. Baseline", self.baseline, BASELINES),
        ]):
            ttk.Label(choices, text=label).grid(row=0, column=col, sticky="w")
            combo = ttk.Combobox(choices, textvariable=variable, values=values,
                                 state="readonly", width=28)
            combo.grid(row=1, column=col, padx=(0, 12))
            self.selectors.append(combo)
        self.selectors[0].bind("<<ComboboxSelected>>", self.change_domain)
        self.selectors[2].bind("<<ComboboxSelected>>", self.refresh)
        params = ttk.Frame(frame)
        params.pack(fill="x")
        self.entries = []
        for label, variable in [("Seed (빈칸: 무작위)", self.seed), ("Max steps", self.max_steps)]:
            ttk.Label(params, text=label).pack(side="left", padx=(0, 8))
            entry = ttk.Entry(params, textvariable=variable, width=12)
            entry.pack(side="left", padx=(0, 20))
            self.entries.append(entry)
        ttk.Label(frame, textvariable=self.note, wraplength=990).pack(anchor="w", pady=10)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        self.preview_button = ttk.Button(buttons, text="명령 미리보기", command=self.preview)
        self.preview_button.pack(side="left")
        self.run_button = ttk.Button(buttons, text="한 번 실행", command=self.run)
        self.run_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(buttons, text="중지", command=self.stop)
        self.stop_button.pack(side="left", padx=8)
        ttk.Label(buttons, textvariable=self.status).pack(side="right")
        self.output = scrolledtext.ScrolledText(frame, wrap="word", state="disabled", font=(self.ui_font, 12), padx=10, pady=10, spacing1=2, spacing3=2)
        self.output.pack(fill="both", expand=True, pady=(12, 0))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()
        self.after(100, self.drain)

    def configure_fonts(self):
        families = set(tkfont.families(self))
        self.ui_font = next((name for name in ("Noto Sans CJK KR", "NanumGothic", "Noto Sans", "DejaVu Sans")
                             if name in families), "sans-serif")
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkFixedFont"):
            tkfont.nametofont(name).configure(family=self.ui_font, size=12)
        self.option_add("*TCombobox*Listbox.font", (self.ui_font, 12))
        style = ttk.Style(self)
        style.configure("TLabel", font=(self.ui_font, 12))
        style.configure("TButton", font=(self.ui_font, 12), padding=(10, 6))
        style.configure("TCombobox", font=(self.ui_font, 12), padding=4)
        style.configure("TEntry", font=(self.ui_font, 12), padding=4)

    def change_domain(self, _event=None):
        values = scenes(self.domain.get())
        self.selectors[1].configure(values=values)
        if self.scene.get() not in values:
            self.scene.set(values[0] if values else "")

    def refresh(self, _event=None):
        busy = self.process is not None
        self.note.set("선택한 scene을 1회 실행합니다. 피드백: auto oracle. 출력에 로그 경로가 표시됩니다."
                      + ("\nIntroPlan: BRL 추론 예시 검색과 설명 생성 후 행동 점수화. 기본 qhat: 고정 후보 데이터, 목표 coverage 95% 보정값입니다."
                         if self.baseline.get() == "introplan" else ""))
        for combo in self.selectors:
            combo.configure(state="disabled" if busy else "readonly")
        for entry in self.entries:
            entry.configure(state="disabled" if busy else "normal")
        self.run_button.configure(state="disabled" if busy else "normal")
        self.preview_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")

    def append(self, text):
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def launch_spec(self):
        command, overrides = build_launch(self.domain.get(), self.scene.get(), self.baseline.get(),
                                          self.seed.get(), self.max_steps.get())
        # Freeze a generated seed so preview and the subsequent run match.
        self.seed.set(overrides["SEED"])
        return command, overrides

    def preview(self):
        try:
            command, overrides = self.launch_spec()
        except ValueError as exc:
            messagebox.showerror("실행 설정", str(exc))
            return
        self.append("\n" + shlex.join([f"{k}={v}" for k, v in overrides.items()] + command) + "\n")

    def run(self):
        if self.process is not None:
            return
        try:
            command, overrides = self.launch_spec()
        except ValueError as exc:
            messagebox.showerror("실행 설정", str(exc))
            return
        self.start(command, PROJECT_ROOT, overrides)

    def start(self, command, cwd, overrides):
        self.append("\n$ " + shlex.join(command) + "\n")
        if overrides:
            self.append(" ".join(f"{k}={v}" for k, v in overrides.items()) + "\n")
        env = os.environ.copy()
        env.update(overrides)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors="replace", bufsize=1, start_new_session=True)
        except OSError as exc:
            messagebox.showerror("실행 실패", str(exc))
            return
        self.process = process
        self.status.set("실행 중")
        self.refresh()
        threading.Thread(target=self.read_output, args=(process,), daemon=True).start()

    def read_output(self, process):
        try:
            for line in process.stdout:
                self.events.put(("output", line))
        finally:
            process.stdout.close()
            self.events.put(("exit", process.wait()))

    def drain(self):
        # Limit each batch so verbose output cannot starve the GUI's stop button.
        for _ in range(300):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "output":
                self.append(value)
            else:
                self.append(f"\n[프로세스 종료: {value}]\n")
                self.status.set("완료" if value == 0 else f"종료 코드 {value}")
                self.process = None
                self.refresh()
                if self.closing:
                    self.destroy()
                    return
        self.after(100, self.drain)

    def stop(self):
        if self.process is None:
            return
        process = self.process
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.status.set("중지 중")
        self.after(2000, lambda: self.kill_remaining(process))

    @staticmethod
    def kill_remaining(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def close(self):
        if self.process is not None:
            self.closing = True
            self.stop()
        else:
            self.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    BaselineGui().mainloop()


if __name__ == "__main__":
    main()
