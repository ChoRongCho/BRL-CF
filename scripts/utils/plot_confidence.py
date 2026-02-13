import matplotlib.pyplot as plt
import os

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
