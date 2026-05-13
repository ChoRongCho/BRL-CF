
class Prompt:
    def __init__(self):
        pass

    def refining_query_prompt(self, target_fact, action_name=None):
        return [
            (
                "system",
                "You convert symbolic blocksworld facts into one concise natural-language yes/no question. "
                "Ask only the question. Do not explain.",
            ),
            (
                "human",
                f"Action: {action_name or 'unknown'}\n"
                f"Fact: {target_fact}\n\n"
                "Predicate meanings:\n"
                "- on(block1, block2): block1 is on block2.\n"
                "- ontable(block): the block is on the table.\n"
                "- clear(block): nothing is on top of the block.\n"
                "- holding(block): the robot is holding the block.\n"
                "- handempty(): the robot hand is empty.\n\n"
                "Question:",
            ),
        ]
