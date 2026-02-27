import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from scripts.feedback_manager import GT_MODEL_CONFIDENCE


PREDICATES = list(GT_MODEL_CONFIDENCE.keys())


def plot_history(history, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)

    for pred, series in history.items():
        plt.figure(figsize=(8, 5))

        plt.plot(series, label=f"{pred} (KB)")
        plt.axhline(
            GT_MODEL_CONFIDENCE[pred],
            linestyle="--",
            color="red",
            label=f"{pred} GT={GT_MODEL_CONFIDENCE[pred]:.2f}"
        )

        plt.ylim(0.0, 1.0)
        plt.xlabel("timestep")
        plt.ylabel("confidence")
        plt.title(f"Confidence over time: {pred}")
        plt.legend()
        plt.tight_layout()

        filepath = os.path.join(save_dir, f"{pred}_confidence.png")
        plt.savefig(filepath, dpi=300)   # ← 저장
        plt.close()                      # 메모리 정리

        print(f"[SAVED] {filepath}")




def save_as_json(query_rate, cls_stats, logs, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_log_{timestamp}.json"

    data = {
        "query_rate": query_rate,
        "cls_stats": cls_stats,
        "logs": logs,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"[SAVED] JSON -> {filename}")
    
