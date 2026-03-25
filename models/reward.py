from models.state import State, get_state, get_types
from models.action import Action, Grounding, ActionSchema, get_actions
from models.observation import ObservationModel, Observation
from models.transition import TransitionModel


class RewardModel:
    def __init__(self, domain_name, goal):
        self.domain_name = domain_name
        self.goal = goal
        
    
    
    def get_reward(self, state: State, action: Action, next_state: State):
        return 1.0