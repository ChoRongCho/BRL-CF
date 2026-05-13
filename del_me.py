from pathlib import Path

name = Path("logs/tomato/scene_metrics/step_50_thres_0-9")
a = name.name.removeprefix("step_50_thres_").replace("-", ".")
print(float(a))