from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


BASELINE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASELINE_DIR / "data"

MOBILE_SCENARIO_FILE = DATA_DIR / "metabot-tasks-info.txt"
MOBILE_MC_PROMPT_FILE = DATA_DIR / "metabot-mc-gen-prompt.txt"
WASTE_MC_PROMPT_FILE = DATA_DIR / "waste-mc-gen-prompt.txt"
TOMATO_MC_PROMPT_FILE = DATA_DIR / "tomato-mc-gen-prompt.txt"


def ensure_mobile_dataset() -> Tuple[Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MOBILE_SCENARIO_FILE.exists() and MOBILE_MC_PROMPT_FILE.exists():
        return MOBILE_SCENARIO_FILE, MOBILE_MC_PROMPT_FILE

    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "gdown", "--pre"])

    import gdown

    if not MOBILE_SCENARIO_FILE.exists():
        gdown.download(
            "https://drive.google.com/uc?id=1XWIeGfF08V1eR104VLDilmwhIGVk2uzk",
            str(MOBILE_SCENARIO_FILE),
            quiet=False,
        )
    if not MOBILE_MC_PROMPT_FILE.exists():
        gdown.download(
            "https://drive.google.com/uc?id=1iEIZaVbbajMXsNdrjVkOgK5rhPtfl5WI",
            str(MOBILE_MC_PROMPT_FILE),
            quiet=False,
        )
    return MOBILE_SCENARIO_FILE, MOBILE_MC_PROMPT_FILE


def load_mobile_dataset(num_calibration_data: int = 200, num_test_data: int = 100) -> Tuple[List[Dict], List[Dict]]:
    scenario_path, prompt_path = ensure_mobile_dataset()
    first_text = scenario_path.read_text(encoding="utf-8")
    second_text = prompt_path.read_text(encoding="utf-8")

    if "--0000--" in first_text and "--0000--" not in second_text:
        prompt_text = first_text
        scenario_text = second_text
    else:
        scenario_text = first_text
        prompt_text = second_text

    scenario_info_text = [item.strip() for item in scenario_text.split("\n\n") if item.strip()]
    mc_gen_prompt_all = [item.strip() for item in prompt_text.split("--0000--") if item.strip()]
    dataset_size = min(len(scenario_info_text), len(mc_gen_prompt_all))
    requested_size = num_calibration_data + num_test_data

    if requested_size > dataset_size:
        raise ValueError(
            f"Requested {requested_size} mobile samples, but only {dataset_size} paired samples are available."
        )

    calibration_set = [
        {"info": scenario_info_text[i], "mc_gen_prompt": mc_gen_prompt_all[i]}
        for i in range(num_calibration_data)
    ]
    test_set = [
        {"info": scenario_info_text[i], "mc_gen_prompt": mc_gen_prompt_all[i]}
        for i in range(num_calibration_data, num_calibration_data + num_test_data)
    ]
    return calibration_set, test_set
