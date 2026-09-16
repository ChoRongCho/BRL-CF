# KnowNo Calibration Summary

- Generated: 2026-06-24 20:03:07
- Log root: `/home/changmin/PyProject/00_BRL-CF/experiments_logs/calibration_log`
- Targets: 95%, 85%, 75% success
- Calibration size: number of calibration examples used to compute qhat

## Calibration Datasets

The calibration records are loaded by `scripts/baseline/compute_qhat.py` from the domain-specific files in `scripts/baseline/data/`.

| Domain | Source directory | Prompt file | Info file | Calibration size |
| --- | --- | --- | --- | --- |
| tomato | `scripts/baseline/data` | `scripts/baseline/data/tomato-mc-gen-prompt.txt` | `scripts/baseline/data/tomato-tasks-info.txt` | 100 |
| wastesorting | `scripts/baseline/data` | `scripts/baseline/data/waste-mc-gen-prompt.txt` | `scripts/baseline/data/waste-tasks-info.txt` | 100 |

## Successful Runs

| Domain | Model | Calibration size | Temp. | qhat@95% | qhat@85% | qhat@75% | Run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tomato | gpt-3.5-turbo | 100 | 5.0 | 0.9243 | 0.9082 | 0.8938 | `experiments_logs/calibration_log/tomato/gpt35turbo/20260624_152140` |
| tomato | gpt-4o | 100 | 5.0 | 0.8404 | 0.7779 | 0.7322 | `experiments_logs/calibration_log/tomato/gpt4/20260624_151751` |
| wastesorting | gpt-3.5-turbo | 100 | 5.0 | 0.9028 | 0.8851 | 0.8512 | `experiments_logs/calibration_log/wastesorting/gpt35turbo/20260624_152844` |
| wastesorting | gpt-4o | 100 | 5.0 | 0.8704 | 0.7369 | 0.7084 | `experiments_logs/calibration_log/wastesorting/gpt4/20260624_152507` |

## Failed Runs

| Domain | Model | Error type | Error | Run |
| --- | --- | --- | --- | --- |
| tomato | PaLM-2L | RuntimeError | LLM API error: Error code: 404 - {'error': {'message': 'The model `PaLM-2L` does not exist or you do not have access to it.', 'type': 'invalid_request_error', 'param': None, 'code' | `experiments_logs/calibration_log/tomato/palm2l/20260624_142101` |
| wastesorting | PaLM-2L | RuntimeError | LLM API error: Error code: 404 - {'error': {'message': 'The model `PaLM-2L` does not exist or you do not have access to it.', 'type': 'invalid_request_error', 'param': None, 'code' | `experiments_logs/calibration_log/wastesorting/palm2l/20260624_142109` |
