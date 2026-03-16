from pathlib import Path
# from utils.asp import DomainRuleBridge
from environments.env import Environment
from utils.arguments import parse_args
from models.belief_state import Belief


def main():
    # domains = ["tomato", "blocksworld", "wastesorting"]
    domains = ["tomato"]
    
    
    for dom in domains:
        args = parse_args(dom)
        env = Environment(args)
        b = Belief()
        w = b.generate_possible_worlds(env.asp_program)
        w.print_atoms()

    
    
if __name__ == "__main__":
    main()
