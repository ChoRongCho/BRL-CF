from pathlib import Path
from utils.asp_bridge import DomainRuleBridge




def main():
    bridge = DomainRuleBridge("configs/blocksworld_domain_rule.yaml")
    bridge.load()

    bridge.add_runtime_facts([
        "block(b1)",
        "block(b2)",
        "block(b3)",
        "block(b4)",
        "block(b5)",
        "block(b6)",
        "block(b7)",
        "support(table)",
        "observed_on(b1,table)",
        "observed_on(b2,b1)",
        "observed_on(b3,table)",
    ])

    bridge.add_runtime_rules([
        "support(B) :- block(B)"
    ])

    worlds = bridge.solve(max_models=0)
    print(bridge.build_program())
    
    print(f"number of possible worlds = {len(worlds)}")
    for w in worlds[:5]:
        print(f"\nWORLD {w.world_id}")
        for atom in w.atoms:
            print(atom)
        
    
if __name__ == "__main__":
    main()
