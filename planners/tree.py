"""
planners/tree.py

Refactored auxiliary code for POMCP-style tree search.
"""



from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from numpy.random import binomial, choice
from models.state import State
from models.action import Action
from models.observation import ObservationModel, Observation


@dataclass
class Node:
    id: int
    parent_id: Optional[int]
    node_type: str  # "observation" or "action"

    # edge_key:
    # - observation 노드일 경우: 선택한 'action'이 key가 됨
    # - action 노드일 경우: 받은 'observation'이 key가 됨
    children: Dict[Any, int] = field(default_factory=dict)

    # statistics
    visits: int = 0
    value: float = 0.0  # 평균 보상
    
    # POMCP
    knowledge: Optional[State] = None
    frontiers: List[State] = field(default_factory=list)
    
    def is_leaf(self) -> bool:
        return self.visits == 0
    
    @property
    def is_action_node(self) -> bool:
        return self.node_type == "action"
    
    @property
    def is_observation_node(self) -> bool:
        return self.node_type == "observation"


class POMDPTree:
    def __init__(self):
        self.nodes = {}
        self.next_id = 0
        
        # root 생성
        self.root_id = self.add_node(None, "observation")
        self.history = []
    
    
    def add_node(self, parent_id, node_type, edge_key=None) -> int:
        new_id = self.next_id
        
        new_node = Node(
            id=new_id,
            parent_id=parent_id,
            node_type=node_type
        )
        
        self.nodes[new_id] = new_node
        
        if parent_id is not None:
            self.nodes[parent_id].children[edge_key] = new_id
        
        self.next_id += 1
        return new_id
        
        
    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def get_history(self):
        return self.history

    def is_leaf_node(self, node_id: int) -> bool:
        node = self.get_node(node_id)
        return node.is_leaf()




    def expand_tree_from(self, parent: int, edge, is_action: bool = False) -> int:
        """
        parent 아래에 child node를 추가한다.
        observation node 아래 action child를 만들 때 is_action=True,
        action node 아래 observation child를 만들 때 is_action=False.
        이미 같은 edge_key가 있으면 기존 child id를 반환한다.
        """
        if parent not in self.nodes:
            raise KeyError(f"Parent node {parent} does not exist.")

        if is_action:
            if isinstance(edge, Action):
                edge_key = edge.name
            else:
                edge_key = str(edge)
            node_type = "action"
            
        else:
            # observation은 문자열, 튜플 등 hashable한 값으로 변환
            if isinstance(edge, list):
                edge_key = tuple(edge)
            else:
                edge_key = edge
            node_type = "observation"
        
        if edge_key in self.nodes[parent].children:
            return self.nodes[parent].children[edge_key]

        return self.add_node(parent_id=parent, node_type=node_type, edge_key=edge_key)







    def increment_visit(self, node_id: int, amount: int = 1) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} does not exist.")
        self.nodes[node_id].visits += amount

    def set_value_if_first(self, node_id: int, value: float) -> None:
        """
        보통 leaf에서 rollout 값으로 최초 초기화할 때 사용.
        increment_visit 이후 호출된다고 가정.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} does not exist.")

        node = self.nodes[node_id]
        if node.visits == 1:
            node.value = value

    # def get_observation_node(self, action_node_id: int, observation: Any) -> int:
    #     """
    #     action node 아래 observation edge를 따라 child를 가져오거나,
    #     없으면 새로 observation node를 생성한다.
    #     """
    #     if action_node_id not in self.nodes:
    #         raise KeyError(f"Action node {action_node_id} does not exist.")

    #     action_node = self.nodes[action_node_id]

    #     if not action_node.is_action_node:
    #         raise TypeError(
    #             f"Node {action_node_id} is not an action node. "
    #             f"Current type: {action_node.node_type}"
    #         )

    #     if observation in action_node.children:
    #         return action_node.children[observation]

    #     return self.add_node(
    #         parent_id=action_node_id,
    #         node_type="observation",
    #         edge_key=observation
    #     )


    def get_observation_node(self, action_node_id: int, observation) -> int:
        obs_key = self._make_obs_key(observation)
        return self.expand_tree_from(action_node_id, obs_key, is_action=False)

    
    def update_action_value(self, action_node_id: int, total_return: float) -> None:
        """
        action node의 평균 value를 incremental mean으로 갱신한다.
        increment_visit(action_node_id) 이후 호출된다고 가정.
        """
        if action_node_id not in self.nodes:
            raise KeyError(f"Action node {action_node_id} does not exist.")

        node = self.nodes[action_node_id]

        if not node.is_action_node:
            raise TypeError(
                f"Node {action_node_id} is not an action node. "
                f"Current type: {node.node_type}"
            )

        if node.visits <= 0:
            raise ValueError(
                f"Cannot update value for node {action_node_id} because visits={node.visits}. "
                f"Call increment_visit first."
            )

        node.value += (total_return - node.value) / node.visits
        
    def _make_obs_key(self, observation: Observation):
        return tuple(sorted(observation.facts.facts))