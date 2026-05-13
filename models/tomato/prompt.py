

class Prompt:
    def __init__(self):
        pass

    def refining_query_prompt(self, target_fact, action_name=None):
        return [
            (
                "system",
                "You convert symbolic tomato-harvesting facts into one concise natural-language yes/no question. "
                "Ask only the question. Do not explain.",
            ),
            (
                "human",
                f"Action: {action_name or 'unknown'}\n"
                f"Fact: {target_fact}\n\n"
                "Predicate meanings:\n"
                "- at(tomato, stem): the tomato is located at the stem.\n"
                "- ripe(tomato): the tomato is ripe.\n"
                "- unripe(tomato): the tomato is unripe.\n"
                "- rotten(tomato): the tomato is rotten.\n"
                "- holding(robot, tomato): the robot is holding the tomato.\n"
                "- handempty(robot): the robot hand is empty.\n\n"
                "Question:",
            ),
        ]
