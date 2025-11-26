from src.config.enums import Direction, StateMember
from src.game.screen import Screen
from src.actors import Snake, Food

import numpy as np
from dataclasses import dataclass, field
import xgboost
import mlflow.sklearn
import mlflow.xgboost
import os


@dataclass(kw_only=True)
class GameState:
    screen: Screen
    snakes: list[Snake]
    food: Food
    ai_model_mlflow_path_or_uri: str
    last_input_direction: Direction = Direction.RIGHT
    current_ai_snake: int = 0
    ai_model: xgboost.sklearn.XGBRegressor = xgboost.sklearn.XGBRegressor()
    ai_model_inference_times: list[float] = field(
        default_factory=lambda: [0.0]
    )

    @property
    def filled_positions(self) -> list[tuple[int, int]]:
        filled_positions = [
            _pos for _snake in self.snakes for _pos in _snake.body
        ]
        return filled_positions

    @property
    def free_positions(self) -> list[tuple[int, int]]:
        free_positions = [
            _pos
            for _pos in self.screen.all_positions
            if _pos not in self.filled_positions
        ]
        return free_positions

    @property
    def board_state(
        self,
    ) -> tuple[dict[tuple[int, int], StateMember], list[StateMember]]:
        _board_state = {
            (i, j): StateMember.EMPTY
            if (
                (i != 0)
                and (j != 0)
                and (i <= self.screen.n_rows)
                and (j <= self.screen.n_cols)
            )
            else StateMember.EDGE
            for i in range(0, self.screen.n_rows + 2)
            for j in range(0, self.screen.n_cols + 2)
        }
        for _pos in self.snakes[0].body:
            _board_state[_pos[0] + 1, _pos[1] + 1] = StateMember.SNAKE0
        for _pos in self.snakes[1].body:
            _board_state[_pos[0] + 1, _pos[1] + 1] = StateMember.SNAKE1

        _board_state[
            self.snakes[0].body[0][0] + 1,
            self.snakes[0].body[0][1] + 1,
        ] = StateMember.SNAKE0_HEAD
        _board_state[
            self.snakes[1].body[0][0] + 1,
            self.snakes[1].body[0][1] + 1,
        ] = StateMember.SNAKE1_HEAD

        _board_state[self.food.position[0] + 1, self.food.position[1] + 1] = (
            StateMember.FOOD
        )

        _state = [
            _board_state[(i, j)]
            for j in range(self.screen.n_cols + 2)
            for i in range(self.screen.n_rows + 2)
        ]
        return _board_state, _state

    def load_ai_model(self) -> None:
        os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
        try:
            self.ai_model = mlflow.xgboost.load_model(
                self.ai_model_mlflow_path_or_uri
            )
        except:
            try:
                self.ai_model = mlflow.sklearn.load_model(
                    self.ai_model_mlflow_path_or_uri
                )
            except Exception as e:
                raise e

    def distance_to_food(self, _snake: Snake) -> float:
        return abs(self.food.position[0] - _snake.body[0][0]) + abs(
            self.food.position[1] - _snake.body[0][1]
        )

    def distance_to_centre(self, _snake: Snake) -> float:
        return abs(10 - _snake.body[0][0]) + abs(10 - _snake.body[0][1])

    def distance_to_wall(self, _snake: Snake) -> float:
        snake_x, snake_y = _snake.body[0]
        distance_x = min(snake_x + 1, self.screen.n_cols - snake_x)
        distance_y = min(snake_y + 1, self.screen.n_rows - snake_y)

        return min(distance_x, distance_y)

    def distances_from_head_to_other_snake(
        self, _snake: Snake, _other_snake: Snake
    ) -> list[tuple[int, int]]:
        _deltas_of_head_to_other_snake = [
            (
                _other_snake_segment_pos[0] - _snake.body[0][0],
                _other_snake_segment_pos[1] - _snake.body[0][1],
            )
            for _other_snake_segment_pos in _other_snake.body
        ]
        return _deltas_of_head_to_other_snake

    def closest_other_snake_segment(
        self, distances_from_head_to_other_snake: list[tuple[int, int]]
    ) -> int:
        _manhattan_distances_to_other_snake_segments = [
            abs(_delta[0]) + abs(_delta[1])
            for _delta in distances_from_head_to_other_snake
        ]
        _closest_segment_index = int(
            np.array(_manhattan_distances_to_other_snake_segments).argmin()
        )
        return _closest_segment_index

    def distance_to_closest_snake_segment(
        self, _snake: Snake, _other_snake: Snake
    ) -> int:
        _distances_of_head_to_other_snake = (
            self.distances_from_head_to_other_snake(_snake, _other_snake)
        )
        _closest_other_snake_segment = self.closest_other_snake_segment(
            _distances_of_head_to_other_snake
        )
        _distance_to_closest_snake_segment = _distances_of_head_to_other_snake[
            _closest_other_snake_segment
        ]
        manhattan_distance_to_closest_snake_segment = abs(
            _distance_to_closest_snake_segment[0]
        ) + abs(_distance_to_closest_snake_segment[1])

        return manhattan_distance_to_closest_snake_segment

    def increment_current_ai_index(self) -> None:
        self.current_ai_snake = (self.current_ai_snake + 1) % 2
