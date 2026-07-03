# Domain Assets

`scripts/domain`은 planner와 simulator가 사용할 도메인 정의 YAML을 모아둔 디렉터리다. 각 도메인은 같은 형태의 파일을 가진다.

- `domain_rule.yaml`: predicate 목록, ASP show directive, 도메인 수준 constraint를 정의한다.
- `robot_skill.yaml`: action schema를 정의한다. Grounder는 여기 적힌 parameter type을 기반으로 가능한 grounded action을 만든다.
- `scene_XX.yaml`: object 목록, 초기 known facts, hidden true facts, symbolic goal을 정의한다.

현재 도메인 asset은 모두 symbolic fact 기반이다. `fluents`, `true_fluents`, `goal_fluents` 같은 numeric fluent 섹션은 여기서는 사용하지 않는다.

## YAML 구성 의미

`facts`는 planner가 처음부터 알고 있는 정보다. 실행 state와 belief state의 출발점이 된다.

`true_init`은 실제 환경의 hidden ground truth다. 부분 관측 도메인에서는 robot이 처음부터 모르는 사실이 여기에 들어간다. Observation model은 sensing action을 실행할 때 이 hidden truth를 참조해 실제 환경처럼 observation을 만든다.

`goal`은 planner가 최종적으로 만족시켜야 하는 symbolic fact 집합이다. 대부분의 reward model은 goal fact들이 새로 만족되는 순간 goal reward를 준다.

부분 관측, 즉 PO(partially observable)는 보통 `facts`에는 없고 `true_init`에는 있는 정보에서 발생한다. 예를 들어 tomato의 ripe/rotten 여부, kitchen의 재료 위치와 청결 상태, rover의 실제 traversable edge, watering의 basket/plant 위치가 PO 요소다.

## 도메인별 설명

### `tomato`

`tomato`는 토마토 수확 도메인이다. Robot은 stem 위치로 이동하고, camera-like detect로 토마토를 관찰한 뒤, 토마토를 집고 scan해서 품질을 확인한다. 정상적으로 수확할 토마토는 `loaded(T,R)`로 싣고, 썩은 토마토는 `discarded(T)`로 버린다. 덜 익은 토마토는 원래 stem에 남겨두는 것이 목표에 포함될 수 있다.

Scene 01 기준 goal은 다음과 같은 의미다.

- 모든 tomato를 `observed(T)` 상태로 만든다.
- ripe tomato인 `tomato1`, `tomato3`을 robot에 load한다.
- rotten tomato인 `tomato2`를 discard한다.
- unripe tomato인 `tomato4`는 `stem_02`에 남긴다.

주요 action은 `navigate`, `detect`, `pick`, `scan`, `place`, `discard`다. `detect`는 stem에 어떤 tomato가 있는지와 겉보기 ripe/unripe 정보를 알려준다. `scan`은 손에 든 tomato가 실제 ripe인지 rotten인지 확인한다.

PO 요소는 토마토의 위치와 품질이다. 초기 `facts`에는 robot 위치, stem, tomato object만 알려져 있고, `true_init`에 `ripe`, `rotten`, `unripe`, `at(T,S)`가 들어 있다. 따라서 planner는 detect/scan observation을 통해 숨겨진 world를 좁혀야 한다.

특징적으로 tomato observation model은 실제 환경처럼 `true_init`을 ground truth로 쓰되, belief update likelihood에서는 candidate state를 기준으로 observation probability를 계산한다. 또한 이미 집거나 load/discard된 tomato는 다시 stem에서 detect되지 않도록 처리한다.

### `wastesorting`

`wastesorting`은 쓰레기 분류 도메인이다. Robot은 pile 위의 waste를 detect하고, 각 waste의 category를 추정한 뒤, 해당 category에 맞는 bin으로 옮긴다.

Scene 01 기준 goal은 다음과 같다.

- `waste1`은 paper bin `pp_bin`에 넣는다.
- `waste2`는 general waste bin `gw_bin`에 넣는다.
- `waste3`은 plastic bin `pl_bin`에 넣는다.
- `waste4`는 can bin `ca_bin`에 넣는다.

주요 action은 `detect_waste`, `pick`, `place_gw_bin`, `place_paper_bin`, `place_can_bin`, `place_plastic_bin`이다. `detect_waste`는 waste가 보이는지와 category label 후보를 관찰하고, `pick`은 detected waste를 집으며, place action은 해당 bin에 넣는다.

PO 요소는 각 waste의 category다. 초기 `facts`에는 waste object와 bin object만 있고, `true_init`에 `paper(W)`, `general(W)`, `plastic(W)`, `can(W)`가 들어 있다. Robot은 detect observation을 통해 category를 추정해야 한다.

특징적으로 detect는 category classification noise를 가진다. Observation model은 실제 category를 기준으로 correct/wrong label 확률을 만들고, transition model은 detected/category fact를 belief state에 반영한다. 이미 bin에 들어갔거나 holding 중인 waste는 다시 detect되지 않도록 막는다.

### `blocksworld`

`blocksworld`는 고전적인 block stacking 도메인이다. 여러 block이 table 위 또는 다른 block 위에 쌓여 있고, robot hand는 한 번에 하나의 block만 들 수 있다. 목표는 초기 tower 배치를 다른 tower 배치로 재구성하는 것이다.

Scene 01 기준 goal은 세 개의 tower를 만드는 것이다.

- 왼쪽 table에는 `b3` 위에 `b2`, 그 위에 `b1`이 오도록 만든다.
- 가운데 table에는 `b6` 위에 `b5`, 그 위에 `b4`가 오도록 만든다.
- 오른쪽 table에는 `b9` 위에 `b8`, 그 위에 `b7`이 오도록 만든다.
- 마지막에는 `handempty`가 참이어야 한다.

주요 action은 `pickup`, `putdown`, `unstack`, `stack`이다. `pickup`은 table 위의 clear block을 들고, `unstack`은 다른 block 위의 clear block을 들어 올린다. `putdown`은 들고 있는 block을 table에 내려놓고, `stack`은 들고 있는 block을 다른 clear block 위에 올린다.

현재 scene에서는 `facts`와 `true_init`이 거의 동일하므로 PO 요소가 약하다. 즉, block 배치가 대부분 알려진 deterministic planning benchmark에 가깝다. 다만 observation field는 action 결과 확인을 위해 존재하며, 이후 scene을 확장하면 일부 `on`, `on_table`, `clear` fact를 hidden으로 옮겨 PO blocksworld로 만들 수 있다.

특징은 상태 전이가 symbolic add/delete effect 중심이라는 점이다. Numeric fluent나 perception label은 없고, action precondition과 effect가 block 관계를 직접 바꾼다.

### `kitchen`

`kitchen`은 고정된 kitchen manipulator가 재료를 찾고, 청결 상태를 확인하고, 필요한 경우 씻은 뒤 soup를 만들어 dish에 serve하는 도메인이다.

Scene 01 기준 goal은 `served_tomato_soup(dish1)`이다. 이를 위해 robot은 tomato와 cheese를 올바른 refrigerator에서 꺼내고, ingredient와 pot/dish가 clean인지 확인하거나 wash한 뒤, pot에 재료를 넣고 `boil_tomato_soup`을 수행한 다음 dish에 serve해야 한다.

주요 action은 다음 흐름으로 나뉜다.

- `open_fridge`: ingredient가 어느 refrigerator에 있는지 관찰한다.
- `pick_ingredient`, `pick_pot`, `pick_dish`: object를 든다.
- `inspect_ingredient`, `inspect_pot`, `inspect_dish`: clean/dirty 상태를 관찰한다.
- `wash_ingredient`, `wash_pot`, `wash_dish`: dirty object를 clean하게 만든다.
- `place_ingredient_in_pot`: clean ingredient를 clean pot에 넣는다.
- `boil_*_soup`: pot 안 재료를 soup로 만든다.
- `serve_*_soup`: cooked soup를 clean dish에 serve한다.

PO 요소는 ingredient refrigerator assignment와 object cleanliness다. 초기 `facts`에는 object type과 robot hand state만 있고, `true_init`에 `in_fridge(I,F)`, `clean(O)`, `dirty(O)`가 들어 있다. Robot은 fridge를 열고 inspect action을 수행해야 hidden fact를 알 수 있다.

특징은 action 수가 많고, object 상태가 여러 단계로 변한다는 점이다. 특히 `clean/dirty`는 inspect 전에는 알 수 없고, dirty라면 wash해야 cooking/serving precondition을 만족할 수 있다.

### `rover`

`rover`는 단일 rover가 waypoint graph를 이동하며 soil sample과 objective image를 수집하고 lander로 data를 communicate하는 도메인이다. IPC rover benchmark를 단순화한 형태다.

Scene 01 기준 goal은 두 가지다.

- `communicated_soil_data(waypoint2)`를 달성한다.
- `communicated_image_data(crater)`를 달성한다.

주요 action은 `navigate`, `detect_road`, `sample_soil`, `take_image`, `communicate_soil_data`, `communicate_image_data`다. Rover는 `can_traverse(W1,W2)`와 `visible(W1,W2)`가 알려진 edge를 따라 이동한다. Soil sample이 있는 waypoint에서 soil analysis를 만들고, objective가 visible한 waypoint에서 image를 찍은 뒤, lander가 보이는 위치에서 data를 communicate한다.

PO 요소는 waypoint graph의 일부 edge, objective visibility, soil sample 위치다. 초기 `facts`에는 일부 `visible`/`can_traverse`만 들어 있고, `true_init`에는 추가 edge, `visible_from(crater,W)`, `at_soil_sample(W)`가 들어 있다. `detect_road`는 숨겨진 road connectivity를 확인하는 observation action이다.

특징은 navigation과 sensing이 결합되어 있다는 점이다. 올바른 길을 발견해야 sample/image 위치까지 이동할 수 있고, data communication은 rover와 lander 사이 visibility가 필요하다.

### `watering`

`watering`은 부분 관측 실내 물주기 도메인이다. Robot은 room graph를 이동하며 basket을 찾고, basket을 집어 tap에서 물을 채운 뒤, plant 위치를 찾아 물을 준다.

Scene 01 기준 goal은 다음과 같다.

- `watered(plant1)`
- `watered(plant2)`

주요 action은 `move`, `find_basket`, `find_plant`, `pick`, `load_water`, `pour_water`다. `move`는 connected room 사이를 이동하고, `find_basket`과 `find_plant`는 현재 room에서 object 위치를 찾는 sensing action이다. `pick`은 basket을 들고, `load_water`는 tap에서 물을 채우며, `pour_water`는 plant가 있는 room에서 물을 준다.

PO 요소는 basket 위치와 plant 위치다. 초기 `facts`에는 room graph와 robot 위치, object type만 있고, `true_init`에 `at(basket1,kitchen)`, `at(plant1,office)`, `at(plant2,living_room)` 같은 실제 위치가 들어 있다. Robot은 `find_basket`, `find_plant`를 통해 hidden location을 알아내야 한다.

특징적으로 basket은 들고 나면 더 이상 방에서 다시 발견되면 안 된다. 모델은 `holding(A,C)` 상태인 basket에 대해 `find_basket` observation/transition을 empty로 처리한다. Plant는 물리적으로 계속 그 방에 존재할 수 있으므로 반복 관찰은 가능하지만, 같은 `at(P,R)` fact가 state에 중복 생성되지는 않는다.

## 명령어

```bash
python3 scripts/domain/scenario_tools.py --mode summary
python3 scripts/domain/scenario_tools.py --mode validate
python3 scripts/domain/scenario_tools.py --mode readme
python3 scripts/domain/scenario_tools.py --mode summary --domain rover
```

`validate`는 YAML parsing, action grounding, ASP loading, numeric-field 미사용 규칙을 검사한다.

## 도메인 추가 규칙

새 도메인을 추가할 때는 최소한 다음 세 파일을 같은 이름의 도메인 디렉터리에 넣는다.

- `domain_rule.yaml`
- `robot_skill.yaml`
- `scene_01.yaml`

`facts`에는 planner가 처음부터 알아도 되는 정보만 넣고, PO로 다룰 정보는 `true_init`으로 분리하는 것이 좋다. Goal은 가능한 한 최종 task 성공을 직접 나타내는 predicate로 작성한다.

## 요약 표

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
