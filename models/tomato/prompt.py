class Prompt:
    def __init__(self):
        pass

    def refining_query_prompt(self, target_fact, action_name=None):
        predicate_descriptions = """
- observed(T): tomato T has been observed.
- scanned(T): tomato T has been scanned.
- ripe(T): tomato T is ripe.
- unripe(T): tomato T is unripe.
- rotten(T): tomato T is rotten.
- holded(T, R): robot R has held tomato T.
- loaded(T, R): tomato T has been loaded by robot R.
- discarded(T): tomato T has been discarded.
- at(T, S): tomato T is at stem/location S.
- handempty(R): robot R's hand is empty.
- holding(R, T): robot R is holding tomato T.
- located(R, L): robot R is located at location L.
""".strip()

        return [
            (
                "system",
                "You convert one symbolic tomato-harvesting fact into exactly one concise Korean yes/no question. "
                "You must preserve the predicate meaning exactly. "
                "Do not change the predicate into another property. "
                "Ask only the final question in Korean."
            ),
            (
                "human",
                f"""
Instruction:
Convert the given symbolic fact into a polite and unambiguous Korean yes/no question.

Important rules:
1. First identify the predicate name in the fact.
2. The question must ask about that predicate only.
3. Do not infer or replace it with another predicate.
4. If the fact is located(R, L), ask whether the robot is at the indicated location.
5. If the fact is ripe(T), unripe(T), or rotten(T), ask about the indicated tomato's state.
6. If the fact is at(T, S), ask whether the indicated tomato is attached to or located at the indicated stem/location.
7. The respondent may not recognize identifiers such as stem_01, stem_02, tomato1, or tomato2.
8. Assume the respondent sees only a tablet display where the relevant object or location is visually indicated.
9. Output only the question.

Examples:
Fact: located(brl_robot,stem_01)
Question: 지금 로봇이 화면에 표시된 위치에 있는 것이 맞나요?

Fact: ripe(tomato1)
Question: 화면에 표시된 토마토가 익은 것이 맞나요?

Fact: rotten(tomato2)
Question: 화면에 표시된 토마토가 상한 것이 맞나요?

Fact: at(tomato1,stem_??)
Question: 현재 줄기에 첫번째 토마토가 달려있나요?

Action: {action_name or 'unknown'}
Fact: {target_fact}

Predicate meanings:
{predicate_descriptions}

Question:
""".strip()
            ),
        ]