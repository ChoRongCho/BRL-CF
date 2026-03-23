from models.state import State, get_state, get_types
from models.action import Action, Grounding, ActionSchema, get_actions
from models.observation import ObservationModel, Observation
from models.transition import TransitionModel
from models.reward import RewardModel

class RewardModel:
    def __init__(self):
        pass
    
    
    def get_reward(state: State, action: Action, next_state: State):
        return