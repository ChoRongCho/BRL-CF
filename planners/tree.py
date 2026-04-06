"""
planners/tree.py

Refactored auxiliary code for POMCP-style tree search.
"""

from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Optional, Tuple

from models.state import State
from models.action import Action
from models.observation import Observation


@dataclass
class Node:
    id: int
    parent_id: Optional[int]
    node_type: str  # "observation" or "action"

    # children: edge_key -> child_id
    children: Dict[Any, int] = field(default_factory=dict)

    # edge_data: edge_key -> original edge object
    # - observation node 아래 action child면 Action 객체 저장
    # - action node 아래 observation child면 obs key 저장
    edge_data: Dict[Any, Any] = field(default_factory=dict)

    visits: int = 0
    value: float = 0.0  # 평균 보상

    # optional belief-related cache
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
        self.nodes: Dict[int, Node] = {}
        self.next_id: int = 0
        self.history: List[Any] = []
        self.root_id: int = self._create_root()

    def _create_root(self) -> int:
        return self.add_node(parent_id=None, node_type="observation")

    def reset(self) -> None:
        self.nodes.clear()
        self.next_id = 0
        self.history = []
        self.root_id = self._create_root()

    def add_node(self, parent_id: Optional[int], node_type: str, edge_key=None, edge_data=None) -> int:
        new_id = self.next_id

        new_node = Node(
            id=new_id,
            parent_id=parent_id,
            node_type=node_type
        )

        self.nodes[new_id] = new_node

        if parent_id is not None:
            parent = self.nodes[parent_id]
            parent.children[edge_key] = new_id
            parent.edge_data[edge_key] = edge_data

        self.next_id += 1
        return new_id

    def get_node(self, node_id: int) -> Node:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} does not exist.")
        return self.nodes[node_id]

    def get_history(self) -> List[Any]:
        return self.history

    def is_leaf_node(self, node_id: int) -> bool:
        return self.get_node(node_id).is_leaf()

    def get_children(self, node_id: int) -> List[int]:
        node = self.get_node(node_id)
        return list(node.children.values())

    def get_child_items(self, node_id: int) -> List[Tuple[Any, int]]:
        """
        return: [(edge_key, child_id), ...]
        """
        node = self.get_node(node_id)
        return list(node.children.items())

    def get_child_edge_data(self, parent_id: int, child_id: int):
        parent = self.get_node(parent_id)
        for edge_key, cid in parent.children.items():
            if cid == child_id:
                return parent.edge_data.get(edge_key)
        raise KeyError(f"Child node {child_id} is not under parent {parent_id}")

    def get_action_children(self, observation_node_id: int) -> List[Tuple[Action, int]]:
        """
        observation node 아래의 action child들을 [(Action, child_id), ...]로 반환
        """
        node = self.get_node(observation_node_id)

        if not node.is_observation_node:
            raise TypeError(
                f"Node {observation_node_id} is not an observation node. "
                f"Current type: {node.node_type}"
            )

        result = []
        for edge_key, child_id in node.children.items():
            edge_obj = node.edge_data.get(edge_key)
            if isinstance(edge_obj, Action):
                result.append((edge_obj, child_id))
        return result

    def get_observation_children(self, action_node_id: int) -> List[Tuple[Any, int]]:
        """
        action node 아래의 observation child들을 [(obs_key, child_id), ...]로 반환
        """
        node = self.get_node(action_node_id)

        if not node.is_action_node:
            raise TypeError(
                f"Node {action_node_id} is not an action node. "
                f"Current type: {node.node_type}"
            )

        result = []
        for edge_key, child_id in node.children.items():
            result.append((edge_key, child_id))
        return result

    def expand_tree_from(self, parent: int, edge, is_action: bool = False) -> int:
        """
        observation node 아래 action child를 만들 때 is_action=True
        action node 아래 observation child를 만들 때 is_action=False

        이미 같은 edge_key가 있으면 기존 child id를 반환한다.
        """
        if parent not in self.nodes:
            raise KeyError(f"Parent node {parent} does not exist.")

        parent_node = self.nodes[parent]

        if is_action:
            if not parent_node.is_observation_node:
                raise TypeError(
                    f"Cannot add an action child under node {parent} "
                    f"because it is not an observation node."
                )

            if isinstance(edge, Action):
                edge_key = self._make_action_key(edge)
                edge_data = edge
            else:
                raise TypeError("Action child expansion requires an Action object.")

            node_type = "action"

        else:
            if not parent_node.is_action_node:
                raise TypeError(
                    f"Cannot add an observation child under node {parent} "
                    f"because it is not an action node."
                )

            edge_key = self._make_obs_key(edge)
            edge_data = edge_key
            node_type = "observation"

        if edge_key in parent_node.children:
            return parent_node.children[edge_key]

        return self.add_node(parent_id=parent, node_type=node_type, 
                             edge_key=edge_key, edge_data=edge_data)


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


    def get_observation_node(self, action_node_id: int, observation) -> int:
        return self.expand_tree_from(action_node_id, observation, is_action=False)


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

    def get_visit(self, node_id: int) -> int:
        return self.get_node(node_id).visits

    def get_value(self, node_id: int) -> float:
        return self.get_node(node_id).value

    def add_particle(self, node_id: int, state: State, max_particles: Optional[int] = None) -> None:
        node = self.get_node(node_id)
        if not node.is_observation_node:
            raise TypeError(f"Node {node_id} is not an observation node.")

        node.frontiers.append(state.copy())
        node.knowledge = state.copy()

        if max_particles is not None and max_particles > 0 and len(node.frontiers) > max_particles:
            node.frontiers = random.sample(node.frontiers, max_particles)

    def sample_particle(self, node_id: int) -> Optional[State]:
        node = self.get_node(node_id)
        if not node.is_observation_node:
            raise TypeError(f"Node {node_id} is not an observation node.")
        if not node.frontiers:
            return None
        return random.choice(node.frontiers).copy()

    def _make_action_key(self, action: Action):
        """
        Action 객체를 tree edge key로 변환.
        문자열 name이 grounding까지 포함한다고 가정.
        """
        return action.name

    def _make_obs_key(self, observation):
        """
        Observation 또는 이미 hashable한 관측 키를 edge key로 변환.
        """
        if isinstance(observation, Observation):
            if hasattr(observation, "state"):
                facts_key = tuple(sorted(observation.state.facts))
                fluents_key = tuple(
                    sorted(
                        (obj, key, float(value))
                        for obj, values in observation.state.fluents.items()
                        for key, value in values.items()
                    )
                )
                return (facts_key, fluents_key)
            if hasattr(observation, "facts") and hasattr(observation.facts, "facts"):
                return tuple(sorted(observation.facts.facts))
            if hasattr(observation, "facts"):
                return tuple(sorted(observation.facts))
            return str(observation)

        if isinstance(observation, list):
            return tuple(observation)

        if isinstance(observation, set):
            return tuple(sorted(observation))

        return observation
    

    def prune(self, node_id: int):
        children = list(self.nodes[node_id].children.values())
        del self.nodes[node_id]

        for child_id in children:
            self.prune(child_id)


    def make_new_root(self, new_root_id: int):
        old_root = self.nodes[new_root_id]

        self.nodes[self.root_id] = Node(
            id=self.root_id,
            parent_id=None,
            node_type="observation",
            children=old_root.children.copy(),
            edge_data=old_root.edge_data.copy(),
            visits=old_root.visits,
            value=old_root.value,
            knowledge=old_root.knowledge,
            frontiers=list(old_root.frontiers),
        )

        if new_root_id != self.root_id:
            del self.nodes[new_root_id]

        for child_id in self.nodes[self.root_id].children.values():
            self.nodes[child_id].parent_id = self.root_id


    def prune_after_action(self, action: Action, observation: Observation):
        action_key = self._make_action_key(action)
        obs_key = self._make_obs_key(observation)

        # root --action--> action node
        action_node_id = self.nodes[self.root_id].children[action_key]

        # action node --observation--> new root
        new_root_id = self.get_observation_node(action_node_id, observation)

        # new_root 브랜치는 삭제되지 않도록 부모에서 잠깐 분리
        del self.nodes[action_node_id].children[obs_key]
        if obs_key in self.nodes[action_node_id].edge_data:
            del self.nodes[action_node_id].edge_data[obs_key]

        # 기존 root 아래 나머지 전부 삭제
        self.prune(self.root_id)

        # 선택된 observation node를 새 root로 승격
        self.make_new_root(new_root_id)

    def debugging(self):
        root = self.get_node(self.root_id)
        observation_nodes = [node for node in self.nodes.values() if node.is_observation_node]
        action_nodes = [node for node in self.nodes.values() if node.is_action_node]
        leaf_nodes = [node for node in self.nodes.values() if node.visits == 0]

        print("\n===== TREE DEBUG =====")
        print(
            f"root={self.root_id} total={len(self.nodes)} "
            f"obs={len(observation_nodes)} act={len(action_nodes)} leaf={len(leaf_nodes)}"
        )
        print(
            f"root_visits={root.visits} root_value={root.value:.4f} "
            f"root_particles={len(root.frontiers)} root_children={len(root.children)}"
        )

        root_knowledge = []
        if root.knowledge is not None:
            root_knowledge = sorted(root.knowledge.facts)
        print(f"root_knowledge={root_knowledge}")

        depth_map: Dict[int, List[int]] = {}
        stack = [(self.root_id, 0)]
        visited = set()
        while stack:
            node_id, depth = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            depth_map.setdefault(depth, []).append(node_id)
            node = self.nodes[node_id]
            for child_id in node.children.values():
                stack.append((child_id, depth + 1))

        print("[Depth Summary]")
        for depth in sorted(depth_map):
            node_ids = depth_map[depth]
            visit_sum = sum(self.nodes[node_id].visits for node_id in node_ids)
            print(f"depth={depth} nodes={len(node_ids)} total_visits={visit_sum}")

        print("[Root Actions]")
        root_actions = self.get_action_children(self.root_id)
        if not root_actions:
            print("  none")
        else:
            ranked_root_actions = sorted(
                root_actions,
                key=lambda item: (
                    self.get_visit(item[1]),
                    self.get_value(item[1]),
                ),
                reverse=True,
            )
            for action, node_id in ranked_root_actions:
                obs_count = len(self.get_observation_children(node_id))
                print(
                    f"  action={action.name} node={node_id} "
                    f"visits={self.get_visit(node_id)} value={self.get_value(node_id):.4f} "
                    f"obs_children={obs_count}"
                )

        print("[Top Visited Nodes]")
        ranked_nodes = sorted(
            self.nodes.values(),
            key=lambda node: (node.visits, node.value),
            reverse=True,
        )
        for node in ranked_nodes[:10]:
            depth = next((d for d, ids in depth_map.items() if node.id in ids), None)
            print(
                f"  node={node.id} type={node.node_type} depth={depth} "
                f"parent={node.parent_id} visits={node.visits} value={node.value:.4f} "
                f"children={len(node.children)} particles={len(node.frontiers)}"
            )

        
