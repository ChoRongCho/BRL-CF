class Prompt:
    def __init__(self):
        pass

    def refining_query_prompt(self, target_fact, action_name=None):
        predicate_descriptions = """
- detected(W): waste item W has been detected.
- plastic(W): waste item W is plastic.
- can(W): waste item W is a can.
- paper(W): waste item W is paper.
- general(W): waste item W is general waste.
- holding(R, W): robot R is holding waste item W.
- handempty(R): robot R's hand is empty.
- in_bin(W, B): waste item W is in bin B.
""".strip()

        return [
            (
                "system",
                "You convert one symbolic waste-sorting fact into exactly one concise Korean yes/no question. "
                "You must preserve the predicate meaning exactly. "
                "Do not change the predicate into another property. "
                "Ask only the final question in Korean.",
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
4. If the fact is detected(W), ask whether the indicated waste item has been detected.
5. If the fact is plastic(W), can(W), paper(W), or general(W), ask about the indicated waste item's type.
6. If the fact is holding(R, W), ask whether the robot is holding the indicated waste item.
7. If the fact is handempty(R), ask whether the robot's hand is empty.
8. If the fact is in_bin(W, B), ask whether the indicated waste item is in the indicated bin.
9. The respondent may not recognize identifiers such as waste1, waste2, robot1, bin_01, or bin_02.
10. Assume the respondent sees only a tablet display where the relevant object, robot, or bin is visually indicated.
11. Output only the question.

Examples:
Fact: detected(waste1)
Question: 화면에 표시된 폐기물이 감지된 것이 맞나요?

Fact: plastic(waste1)
Question: 화면에 표시된 폐기물이 플라스틱인 것이 맞나요?

Fact: holding(robot1,waste2)
Question: 현재 로봇이 폐기물을 들고 있는 것이 맞나요?

Fact: in_bin(waste1,bin_01)
Question: 화면에 표시된 폐기물이 표시된 수거함 안에 있는 것이 맞나요?

Action: {action_name or 'unknown'}
Fact: {target_fact}

Predicate meanings:
{predicate_descriptions}

Question:
""".strip()
            ),
        ]
