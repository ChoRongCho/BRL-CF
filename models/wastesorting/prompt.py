
class Prompt:
    def __init__(self):
        pass

    def refining_query_prompt(self, target_fact, action_name=None):
        return [
            (
                "system",
                "You convert symbolic waste-sorting facts into one concise natural-language yes/no question. "
                "Ask only the question. Do not explain.",
            ),
            (
                "human",
                f"Action: {action_name or 'unknown'}\n"
                f"Fact: {target_fact}\n\n"
                "Predicate meanings:\n"
                "- plastic(waste): the waste item is plastic.\n"
                "- can(waste): the waste item is a can.\n"
                "- paper(waste): the waste item is paper.\n"
                "- general(waste): the waste item is general waste.\n"
                "- in_bin(waste, bin): the waste item is in the bin.\n\n"
                "Question:",
            ),
        ]
