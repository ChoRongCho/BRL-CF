
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
- located(R, L): robot R is located at L.
""".strip()

        return [
            (
                "system",
                "Convert a symbolic tomato-harvesting fact into one concise natural-language yes/no question. "
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
