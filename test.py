import numpy as np
from models.feedback_manager import FeedbackManger
from models.belief import Belief
from models.state import State

def main():
    
    # Test set
    weights = [0.3, 0.11, 0.05, 0.25, 0.1, 0.1, 0.1, 0.05, 0.04]
    b = Belief(
        knowledge=State(['knowledge1', 'knowledge2', 'knowledge3']),
        frontier=[
            State(['knowledge1', 'knowledge3', 'A', 'X', 'i']),
            State(['knowledge1', 'knowledge3', 'B', 'Y', 'i']),
            State(['knowledge1', 'knowledge3', 'C', 'X', 'j']),
            State(['knowledge1', 'knowledge3', 'A', 'Y', 'j']),
            State(['knowledge1', 'knowledge3', 'B', 'X', 'k']),
            State(['knowledge1', 'knowledge3', 'C', 'Y', 'k']),
            State(['knowledge1', 'knowledge2', 'C', 'Y', 'i']),
            State(['knowledge1', 'knowledge2', 'A', 'X', 'j']),
            State(['knowledge1', 'knowledge2', 'B', 'X', 'k'])
            
        ],
        frontier_weights=np.array(weights)
        
    )

    fm = FeedbackManger(0.7)
    
    belief = fm.get_new_observation(belief=b)
    
    





if __name__ == "__main__":
    main()