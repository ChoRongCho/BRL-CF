# Domain Assets

`scripts/domain` contains the YAML assets used to initialize planning domains.
Each domain directory should provide the same three files:

- `domain_rule.yaml`: predicate inventory, ASP show directives, and domain-level constraints.
- `robot_skill.yaml`: action schemas used by the local action grounder.
- `scene_01.yaml`: objects, known facts, hidden/true facts, and symbolic goals.

The current domain assets are symbolic. Numeric fluent sections such as `fluents`,
`true_fluents`, `goal_fluents`, and numeric action-effect fields are intentionally
not used here.

## Commands

```bash
python3 scripts/domain/scenario_tools.py --mode summary
python3 scripts/domain/scenario_tools.py --mode validate
python3 scripts/domain/scenario_tools.py --mode readme
python3 scripts/domain/scenario_tools.py --mode summary --domain rover
```

`validate` checks YAML parsing, action grounding, ASP loading, and the no-numeric-field
rule for all discovered domains.

## Notes

- `tomato` and `wastesorting` are the original task domains.
- `blocksworld`, `kitchen`, `rover`, and `watering` are additional symbolic domain assets.
- Adding a new domain only requires creating a new directory with the three required files above.

<!-- DOMAIN_SUMMARY_START -->

| Domain | Label | Scenes | Types | Objects | Facts | True Init | Goals | Actions | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `blocksworld` | blocksworld | 1 | 2 | 12 | 25 | 13 | 13 | 4 | benchmarks p001/domain.pddl where available |
| `kitchen` | kitchen | 3 | 8 | 11 | 12 | 9 | 1 | 21 | - |
| `rover` | rover | 1 | 4 | 7 | 13 | 19 | 2 | 6 | simplified from benchmarks/32_ROVER_IPC23 p001 |
| `tomato` | TomatoHarvest | 25 | 4 | 10 | 12 | 8 | 8 | 6 | - |
| `wastesorting` | WasteSorting | 25 | 6 | 9 | 10 | 4 | 4 | 6 | - |
| `watering` | watering | 1 | 5 | 13 | 31 | 21 | 2 | 6 | - |

<!-- DOMAIN_SUMMARY_END -->
