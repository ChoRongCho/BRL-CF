# models.reward.py
from models.state import State, get_state, get_types
from models.action import Action, Grounding, ActionSchema, get_actions
from models.observation import ObservationModel, Observation
from models.transition import TransitionModel


class RewardModel:
    def __init__(self, domain_name, goal):
        self.domain = domain_name
        self.goal = goal
        
        self.reward_model = None
        
        self.load_reward_function()
    
    def load_reward_function(self):
        if self.domain == "tomato":
            from models.tomato.rw import RewardTomato
            self.reward_model = RewardTomato(self.goal)
        elif self.domain == "blocksworld":
            from models.blocksworld.rw import RewardBlocksworld
            self.reward_model = RewardBlocksworld(self.goal)
        elif self.domain == "wastesorting":
            from models.wastesorting.rw import RewardWastesorting
            self.reward_model = RewardWastesorting(self.goal)
    
    def get_reward(self, state: State, action: Action, next_state: State):
        
        reward = self.reward_model.calculate_reward(state, action, next_state)
        
        return reward