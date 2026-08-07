# KnowNo Domain Modules

Domain-specific KnowNo code lives here.

- `tomato/`: tomato harvesting helpers and logger.
- `wastesorting/`: waste sorting semantics in `semantics.py`.
- `watering/`, `rover/`, `kitchen/`, `blocksworld/`: placeholder modules for upcoming domain extensions.

Each domain should keep the same layout:

- `semantics.py`: action parsing, hidden-state initialization, validation rules, and domain logger.
- `answer.py`: auto-answer policy for scripted KnowNo help queries.
- `planning_prompts.py`: planning-time option generation and scoring prompts.
- `calibration_prompts.py`: calibration prompt builders and calibration template.
- `calibration_dataset/`: domain-specific calibration prompt/info text files.

Keep shared LLM, prompt parsing, calibration, and runner infrastructure outside this
folder. Put only domain semantics here: action parsing, hidden-state initialization,
prompt builders, validation rules, auto-answer policies, and domain loggers.
