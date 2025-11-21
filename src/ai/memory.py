from collections import namedtuple, deque
import random
from typing import Any

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)


# Using example https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayMemory(object):
    memory: deque

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args: list[Any]) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
