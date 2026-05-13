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
                "Convert a symbolic waste-sorting fact into one concise natural-language yes/no question. "
                "Use the object names exactly as given. Ask only the question.",
            ),
            (
                "human",
                f"Action: {action_name or 'unknown'}\n"
                f"Fact: {target_fact}\n\n"
                f"Predicate meanings:\n{predicate_descriptions}\n\n"
                "Question:",
            ),
        ]
