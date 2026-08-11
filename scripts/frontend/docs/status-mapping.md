# 상태 매핑 정리

## 1. Backend → Frontend 상태 매핑

`websocket.py`의 `_derive_frontend_status()`에서 `environment.status`와 `robot.status`를 조합하여 프론트엔드 상태를 결정한다.

| environment.status | robot.status | → frontend status | playing |
|---|---|---|---|
| `cf_active` | * | (무시, 브로드캐스트 안 함) | - |
| `bt_paused` | * | `halted` | false |
| `idle` | * | `halted` | false |
| `bt_executing` | `*_failed` | (이전 상태 유지) | true |
| `bt_executing` | `preparing_nav` | `preparing_nav` | true |
| `bt_executing` | `navigating` | `navigating` | true |
| `bt_executing` | `scanning` | `scanning` | true |
| `bt_executing` | `picking` | `picking` | true |
| `bt_executing` | `placing` | `placing` | true |
| `bt_executing` | 기타 | robot.status 그대로 | true |
| `task_planning` | * | `task_planning` | false |
| `active` / 기타 | * | `active` | false |

## 2. Frontend 상태 목록 (status_list)

`MainPage3.vue`에서 정의된 프론트엔드 상태 목록.

| label | value | 상태바 메시지 | 활성 버튼 | 비디오 오버레이 |
|---|---|---|---|---|
| 대기중 | `active` | 무엇을 도와드릴까요? | play | 없음 |
| 계획중 | `task_planning` | 로봇이 작업 계획을 세우고 있어요! | (없음) | 로봇이 작업 계획 중입니다! |
| 이동준비중 | `preparing_nav` | 로봇이 이동을 준비중이에요! | play, pause | 전체 화면 오버레이 |
| 로봇이동중 | `navigating` | 로봇이 이동중이에요! | play, pause | 로봇이 이동중이에요 |
| 스캔중 | `scanning` | 로봇이 토마토의 상태를 확인하고 있어요! | pause | 토마토 상태 확인중 |
| 스캔확인중 | `correct_scanning` | 토마토 상태를 확인해주세요! | (없음) | 하단 30% 가이드 |
| 수확진행중 | `picking` | 로봇이 토마토를 수확하고 있어요! | correct, pause | 없음 |
| 수확수정중 | `correct_picking` | 변경하고 싶은 토마토를 클릭해주세요! | (없음) | 하단 30% 가이드 |
| 수확진행중 | `placing` | 로봇이 토마토를 바구니에 넣고 있어요! | correct, pause | 없음 |
| 수확수정중 | `correct_placing` | 이 토마토가 썩었나요? | (없음) | 하단 30% 가이드 |
| 확인요청중 | `correct_proactive` | 토마토 상태를 확인해주세요! | (없음) | 하단 30% ask_class |
| 일시정지 | `halted` | 작업을 계속 하려면 [수확 시작]을 눌러주세요! | play | 없음 |

## 3. 버튼 동작

### 수확시작 (play)
| 현재 상태 | 동작 | WebSocket 명령 |
|---|---|---|
| `active` | BT resume | `start_harvest` → `/bt_pause_resume(false)` |
| `halted` | BT resume | `start_harvest` → `/bt_pause_resume(false)` |
| playing 중 | BT pause | `pause` → `/bt_pause_resume(true)` |

### 도와주기 (correct)
| 현재 상태 | 동작 | WebSocket 명령 |
|---|---|---|
| `picking` | → `correct_picking` | `cf_trigger` |
| `placing` | → `correct_placing` | `cf_trigger` |

### 수확완료 (finish)
| 현재 상태 | 동작 |
|---|---|
| `halted` | 복귀 시뮬레이션 (navigating → active) |

## 4. 페이지별 역할

| 페이지 | 명령 | 설명 |
|---|---|---|
| `IndexPage.vue` | `start_robot` | Task Planning 시작 (최초 1회) |
| `MainPage3.vue` | `start_harvest` / `pause` | BT resume / pause 토글 |
