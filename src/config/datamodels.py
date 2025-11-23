from src.config.enums import Direction

from dataclasses import dataclass


@dataclass(kw_only=True)
class RewardContext:
    previous_direction: Direction
    new_direction: Direction
    previous_distance_to_food: float
    new_distance_to_food: float
    previous_distance_to_wall: float
    new_distance_to_wall: float
    previous_distance_of_head_to_other_snake: int
    new_distance_of_head_to_other_snake: int
    is_food_eaten: bool
    is_game_over: bool
