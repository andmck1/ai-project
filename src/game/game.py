from dataclasses import dataclass
from enum import Enum, unique
import random
from typing import Any
import pygame
import torch
import numpy as np
import pandas as pd
import xgboost

from src.actors import Snake, Food
from src.game.screen import Screen
from src.config.enums import Colour, Direction
# from src.ai.models import DQN

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


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


@unique
class StateMember(Enum):
    EDGE = 0
    EMPTY = 1
    SNAKE0 = 2
    SNAKE0_HEAD = 3
    SNAKE1 = 4
    SNAKE1_HEAD = 5
    FOOD = 6


class Game:
    screen: Screen
    snakes: list[Snake]
    food: Food
    clock: pygame.time.Clock
    framerate: int = 8
    all_positions: list[tuple[int, int]]
    is_two_ai_playing: bool = False
    last_input_direction: Direction = Direction.RIGHT
    is_draw_enabled: bool = True
    current_ai_snake: int = 0
    ai_model: xgboost.sklearn.XGBRegressor | None = None
    is_debug_enabled: bool = True
    is_ai_enabled: bool = True

    def __init__(
        self,
        screen: Screen,
        window_title: str,
        framerate: int = 8,
        is_two_ai_playing: bool = False,
        is_draw_enabled: bool = True,
        is_ai_random: bool = True,
        is_debug_enabled: bool = True,
        is_ai_enabled: bool = True,
    ) -> None:
        self.screen = screen
        self.is_two_ai_playing = is_two_ai_playing
        self.is_draw_enabled = is_draw_enabled

        start_position_1 = (5, 5)

        start_position_2 = (self.screen.n_rows - 5, self.screen.n_cols - 5)

        if is_two_ai_playing:
            self.snakes = [
                Snake(
                    screen=screen,
                    start_position=start_position_1,
                    is_ai=True,
                    name="AI 1",
                    start_direction=Direction.RIGHT,
                ),
                Snake(
                    screen=screen,
                    start_position=start_position_2,
                    is_ai=True,
                    body_colour=Colour.BLUE,
                    name="AI 2",
                    start_direction=Direction.LEFT,
                ),
            ]
        else:
            self.snakes = [
                Snake(
                    screen=screen,
                    start_position=start_position_1,
                    is_ai=False,
                    name="Player",
                    start_direction=Direction.RIGHT,
                ),
                Snake(
                    screen=screen,
                    start_position=start_position_2,
                    is_ai=True,
                    body_colour=Colour.BLUE,
                    name="AI",
                    start_direction=Direction.LEFT,
                ),
            ]

        self.food = Food(screen=screen)
        self.food.move(self.filled_positions)
        self.clock = pygame.time.Clock()
        self.framerate = framerate
        self.all_positions = [
            (i, j)
            for i in range(self.screen.n_cols)
            for j in range(self.screen.n_rows)
        ]
        self.is_ai_random = is_ai_random
        self.is_debug_enabled = is_debug_enabled
        self.is_ai_enabled = is_ai_enabled

        if not is_ai_random:
            self.load_ai_model()

        pygame.init()
        pygame.display.set_caption(window_title)

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
            for _pos in self.all_positions
            if _pos not in self.filled_positions
        ]
        return free_positions

    @property
    def state(
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
            self.snakes[0].body[0][0] + 1, self.snakes[0].body[0][1] + 1
        ] = StateMember.SNAKE0_HEAD
        _board_state[
            self.snakes[1].body[0][0] + 1, self.snakes[1].body[0][1] + 1
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

    def quit(self) -> bool:
        pygame.quit()
        return False

    def handle_events(self) -> bool:
        # Hardcode for single player
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit()
                    return False
                if event.key == pygame.K_UP:
                    self.last_input_direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.last_input_direction = Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    self.last_input_direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.last_input_direction = Direction.RIGHT

        return True

    def random_ai_move(self) -> Direction:
        random_direction: Direction = list(Direction)[
            random.randint(0, len(Direction) - 1)
        ]
        return random_direction

    def load_ai_model(self) -> None:
        # _ai_model = DQN(len(self.state[1]), 4).to(device)
        # _ai_model.load_state_dict(torch.load("models/base_policy_net.pth"))
        _ai_model = xgboost.XGBRegressor()
        _ai_model.load_model("models/xgboost.ubj")
        self.ai_model = _ai_model

    def model_ai_move(self) -> Direction:
        def create_input_row(
            _state: list[Any],
            _reward: float,
            _direction: Direction,
            _column_names: list[str],
        ) -> dict[str | int, int | float]:
            _row: dict[str | int, int | float] = {k: 0 for k in _column_names}
            _row = {k: 0 for k in _column_names}
            for i in range(len(_state) - 5):
                _row[i] = _state[i]

            for _direction_member in list(Direction):
                for _statemember in list(StateMember):
                    _row[
                        f"is_state_at_head_offset_{str(_direction_member.value)}_{str(_statemember.name)}"
                    ] = 0

            for i, _direction_member in zip(
                range(len(_state) - 5, len(_state) - 1), list(Direction)
            ):
                _row[
                    f"is_state_at_head_offset_{str(_direction_member.value)}_{str(_state[i].name)}"
                ] = list(StateMember).index(_state[i])

            _row[f"current_{_state[-1].name}"] = 1
            _row[f"input_{_direction.name}"] = 1
            _row["reward"] = _reward
            return _row

        self.current_ai_snake = 1

        if self.ai_model is not None:
            # state = self.training_ai_get_current_snake_state()
            # state = torch.tensor(
            #     state, dtype=torch.float32, device=device
            # ).unsqueeze(0)
            # prediction = self.ai_model(state).max(1).indices.view(1, 1)
            # return list(Direction)[prediction]
            columns = [
                0,
                1,
                2,
                3,
                4,
                5,
                "input_UP",
                "input_DOWN",
                "input_LEFT",
                "input_RIGHT",
                "current_UP",
                "current_DOWN",
                "current_LEFT",
                "current_RIGHT",
                "reward",
            ]
            state = self.training_ai_get_current_snake_state()
            input_rows = []
            for direction in list(Direction):
                input_row = create_input_row(state, 0, direction, columns)
                input_rows.append(input_row)
            input_rows = pd.DataFrame(input_rows).drop("reward", axis=1)
            direction = list(Direction)[
                self.ai_model.predict(input_rows).argmax()
            ]
            return direction
        else:
            print("Error, unbound AI")
            return Direction.UP

    def ai_move(self) -> Direction:
        if self.is_ai_random:
            return self.random_ai_move()
        else:
            return self.model_ai_move()

    def frame(self) -> bool:
        # TODO - could be more efficient to collect the rectangles
        # and display using .update() for draw calls

        self.screen.surface.fill(Colour.BLACK.value)

        if not self.handle_events():
            return False

        rewardContexts = []

        for _, _snake in enumerate(self.snakes):
            is_food_eaten = False
            is_game_over = False
            previous_direction = _snake.current_direction
            _other_snake = self.snakes[(self.current_ai_snake + 1) % 2]

            previous_distance_to_food = self.distance_to_food(_snake)
            previous_distance_to_wall = self.distance_to_wall(_snake)
            previous_distance_to_other_snake = (
                self.distance_to_closest_snake_segment(_snake, _other_snake)
            )

            if _snake.is_ai:
                _snake.change_direction(self.ai_move())
                if self.is_ai_enabled:
                    _snake.move()
            else:
                _snake.change_direction(self.last_input_direction)
                _snake.move()

            new_distance_to_food = self.distance_to_food(_snake)
            new_distance_to_wall = self.distance_to_wall(_snake)
            new_distance_to_other_snake = (
                self.distance_to_closest_snake_segment(_snake, _other_snake)
            )

            if _snake.body[0] == self.food.position:
                is_food_eaten = True
                _snake.eaten_fruit = True
                self.food.move(self.free_positions)

            if _snake.check_collision(self.filled_positions):
                is_game_over = True
                print(_snake.name + " lost")
                self.quit()

            rewardContext = RewardContext(
                previous_direction=previous_direction,
                new_direction=self.last_input_direction,
                previous_distance_to_food=previous_distance_to_food,
                new_distance_to_food=new_distance_to_food,
                previous_distance_to_wall=previous_distance_to_wall,
                new_distance_to_wall=new_distance_to_wall,
                previous_distance_of_head_to_other_snake=previous_distance_to_other_snake,
                new_distance_of_head_to_other_snake=new_distance_to_other_snake,
                is_game_over=is_game_over,
                is_food_eaten=is_food_eaten,
            )
            rewardContexts.append(rewardContext)

            if is_game_over:
                return False

            if self.is_draw_enabled:
                _snake.draw()

        if self.is_draw_enabled:
            self.food.draw()

        if self.is_debug_enabled:
            self.draw_debug(rewardContexts)

        if self.is_draw_enabled:
            pygame.display.flip()

        self.clock.tick(self.framerate)

        return True

    def draw_debug(self, rewardContexts: list[RewardContext]) -> None:
        _board_state, _ = self.state
        start_pos_x = self.screen.width + self.screen.debug_separation

        # draw title
        start_pos_y = 0
        self.screen.surface.blit(
            self.screen.font.render("State", True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        # draw state
        start_pos_y += self.screen.block_size
        for _pos, _statemember in _board_state.items():
            match _statemember:
                case StateMember.SNAKE0:
                    colour = Colour.GREEN.value
                case StateMember.SNAKE1:
                    colour = Colour.BLUE.value
                case StateMember.FOOD:
                    colour = Colour.RED.value
                case StateMember.EDGE:
                    colour = Colour.YELLOW.value
                case StateMember.SNAKE0_HEAD:
                    colour = Colour.PURPLE.value
                case StateMember.SNAKE1_HEAD:
                    colour = Colour.PURPLE.value
                case _:
                    colour = Colour.GREY.value

            pygame.draw.rect(
                self.screen.surface,
                colour,
                (
                    _pos[0] * self.screen.block_size + start_pos_x,
                    _pos[1] * self.screen.block_size + start_pos_y,
                    self.screen.block_size,
                    self.screen.block_size,
                ),
            )

        # draw title
        start_pos_y += self.screen.block_size * (self.screen.n_rows + 2)
        self.screen.surface.blit(
            self.screen.font.render("AI State", True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        # draw ai snake state
        self.current_ai_snake = 0
        _snake_ai_state = self.training_ai_get_current_snake_state()

        # start_pos_y += self.screen.block_size
        # text = "Snake Pos X: {:.3f}".format(_snake_ai_state[0])
        # self.screen.surface.blit(
        #     self.screen.font.render(text, True, (255, 255, 255)),
        #     (start_pos_x, start_pos_y),
        # )
        #
        # start_pos_y += self.screen.block_size
        # text = "Snake Pos Y: {:.3f}".format(_snake_ai_state[1])
        # self.screen.surface.blit(
        #     self.screen.font.render(text, True, (255, 255, 255)),
        #     (start_pos_x, start_pos_y),
        # )
        #
        start_pos_y += self.screen.block_size
        text = "Distance to Wall X: {:.3f}".format(_snake_ai_state[0])
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        start_pos_y += self.screen.block_size
        text = "Distance to Wall Y: {:.3f}".format(_snake_ai_state[1])
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        start_pos_y += self.screen.block_size
        text = "Fruit Pos X: {:.3f}".format(_snake_ai_state[2])
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        start_pos_y += self.screen.block_size
        text = "Fruit Pos Y: {:.3f}".format(_snake_ai_state[3])
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        start_pos_y += self.screen.block_size
        text = "Closest Segment Pos X: {:.3f}".format(_snake_ai_state[4])
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        start_pos_y += self.screen.block_size
        text = "Closest Segment Pos Y: {:.3f}".format(_snake_ai_state[5])
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        start_pos_y += self.screen.block_size
        text = "State Relative to Head (+1, 0): {}".format(
            _snake_ai_state[6].name
        )
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )
        start_pos_y += self.screen.block_size
        text = "State Relative to Head (-1, 0): {}".format(
            _snake_ai_state[7].name
        )
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )
        start_pos_y += self.screen.block_size
        text = "State Relative to Head (0, +1): {}".format(
            _snake_ai_state[8].name
        )
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )
        start_pos_y += self.screen.block_size
        text = "State Relative to Head (0, -1): {}".format(
            _snake_ai_state[9].name
        )
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        start_pos_y += self.screen.block_size
        text = "Current Direction: {}".format(_snake_ai_state[10].name)
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        # draw rewards
        for i in range(2):
            start_pos_x = 0
            start_pos_y = (
                self.screen.block_size * self.screen.n_rows
                + i * self.screen.block_size
            )
            reward = self.calculate_ai_reward(rewardContexts[i])
            text = f"Snake {i} Reward: {reward:.3f}"
            self.screen.surface.blit(
                self.screen.font.render(text, True, (255, 255, 255)),
                (start_pos_x, start_pos_y),
            )

    def training_ai_frame(self) -> None:
        self.screen.surface.fill(Colour.BLACK.value)

    def training_ai_end_frame(self) -> None:
        if self.is_draw_enabled:
            self.food.draw()
            for _snake in self.snakes:
                _snake.draw()
            pygame.display.flip()

        self.clock.tick(self.framerate)

    def training_ai_get_current_snake_state(self) -> list[Any]:
        state = []
        _snake = self.snakes[self.current_ai_snake]
        _other_snake = self.snakes[(self.current_ai_snake + 1) % 2]
        _board_state, _ = self.state

        _distances_from_head_to_other_snake = (
            self.distances_from_head_to_other_snake(_snake, _other_snake)
        )
        _closest_other_snake_segment = self.closest_other_snake_segment(
            _distances_from_head_to_other_snake
        )
        _distance_to_closest_snake_segment = (
            _distances_from_head_to_other_snake[_closest_other_snake_segment]
        )
        _snake_head_position_x = _snake.body[0][0] + 1
        _snake_head_position_y = _snake.body[0][1] + 1
        _statemembers_around_head = []
        for _direction in list(Direction):
            _statemembers_around_head.append(
                _board_state[
                    (
                        _snake_head_position_x + _direction.value[0],
                        _snake_head_position_y + _direction.value[1],
                    )
                ]
            )

        normalised_distance_to_wall_x = (
            min(_snake.body[0][0], abs(_snake.body[0][0] - self.screen.n_cols))
            / self.screen.n_cols
        )
        normalised_distance_to_wall_y = (
            min(_snake.body[0][1], abs(_snake.body[0][1] - self.screen.n_rows))
            / self.screen.n_rows
        )

        normalised_fruit_position_x_delta = (
            _snake.body[0][0] - self.food.position[0]
        ) / self.screen.n_cols
        normalised_fruit_position_y_delta = (
            _snake.body[0][1] - self.food.position[1]
        ) / self.screen.n_cols

        # _closest_segment_x, _closest_segment_y = _other_snake.body[
        #     _closest_segment_index
        # ]
        # _closest_segment_x /= self.screen.n_cols
        # _closest_segment_y /= self.screen.n_rows

        _closest_segment_x_delta = (
            _distance_to_closest_snake_segment[0] / self.screen.n_cols
        )
        _closest_segment_y_delta = (
            _distance_to_closest_snake_segment[1] / self.screen.n_rows
        )

        # state.append(normalised_position_x)
        # state.append(normalised_position_y)
        state.append(normalised_distance_to_wall_x)
        state.append(normalised_distance_to_wall_y)
        state.append(normalised_fruit_position_x_delta)
        state.append(normalised_fruit_position_y_delta)
        state.append(_closest_segment_x_delta)
        state.append(_closest_segment_y_delta)
        state += _statemembers_around_head
        state.append(_snake.current_direction)

        return state

    # simulate env.step in gym
    # step the game by input direction and return:
    # previous state from current snake point of view,
    # new state from current snake point of view,
    # reward and if game ended
    def training_ai_step(
        self, input_direction: Direction
    ) -> tuple[list[int] | None, float, bool] | None:
        reward = 0.0
        next_state = []
        is_game_over = False
        is_food_eaten = False

        _snake = self.snakes[self.current_ai_snake]
        _other_snake = self.snakes[(self.current_ai_snake + 1) % 2]

        previous_direction = _snake.current_direction

        previous_distance_to_food = self.distance_to_food(_snake)
        previous_distance_to_wall = self.distance_to_wall(_snake)
        previous_distance_to_other_snake = (
            self.distance_to_closest_snake_segment(_snake, _other_snake)
        )

        _snake.change_direction(input_direction)
        _snake.move()

        new_distance_to_food = self.distance_to_food(_snake)
        new_distance_to_wall = self.distance_to_wall(_snake)
        new_distance_to_other_snake = self.distance_to_closest_snake_segment(
            _snake, _other_snake
        )

        if _snake.body[0] == self.food.position:
            is_food_eaten = True
            _snake.eaten_fruit = True
            self.food.move(self.free_positions)

        if _snake.check_collision(self.filled_positions):
            is_game_over = True
            self.quit()

        rewardContext = RewardContext(
            previous_direction=previous_direction,
            new_direction=input_direction,
            previous_distance_to_food=previous_distance_to_food,
            new_distance_to_food=new_distance_to_food,
            previous_distance_to_wall=previous_distance_to_wall,
            new_distance_to_wall=new_distance_to_wall,
            previous_distance_of_head_to_other_snake=previous_distance_to_other_snake,
            new_distance_of_head_to_other_snake=new_distance_to_other_snake,
            is_game_over=is_game_over,
            is_food_eaten=is_food_eaten,
        )
        reward = self.calculate_ai_reward(rewardContext)
        if is_game_over:
            return None, reward, True

        next_state = self.training_ai_get_current_snake_state()
        self.current_ai_snake = (self.current_ai_snake + 1) % 2
        return next_state, reward, is_game_over

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

    def calculate_ai_reward(self, rewardContext: RewardContext) -> float:
        reward = 0.0
        if rewardContext.is_game_over:
            return -1.0
        if rewardContext.is_food_eaten:
            return 1.0
        if rewardContext.new_distance_to_wall == 1:
            return -1.0
        if rewardContext.new_distance_of_head_to_other_snake == 1:
            return -1.0

        if (
            rewardContext.previous_direction.value[0]
            == rewardContext.new_direction.value[0] * -1
        ) and (
            rewardContext.previous_direction.value[1]
            == rewardContext.new_direction.value[1] * -1
        ):
            return -1.0

        food_reward = 0.0
        distance_to_other_snake_reward = 0.0

        distance_moved_toward_food = (
            rewardContext.previous_distance_to_food
            - rewardContext.new_distance_to_food
        )
        distance_moved_toward_other_snake = (
            rewardContext.previous_distance_of_head_to_other_snake
            - rewardContext.new_distance_of_head_to_other_snake
        )

        # DEPENDANT ON MANHATTEN DISTANCE
        # TODO - set all this in training config
        MAX_DIST_FROM_FOOD = self.screen.n_cols + self.screen.n_rows
        MAX_DIST_FROM_OTHER_SNAKE = self.screen.n_cols + self.screen.n_rows
        FOOD_REWARD_MULTIPLIER = 10.0
        DISTANCE_TO_OTHER_SNAKE_MULTIPLIER = 2.0

        food_reward = (
            FOOD_REWARD_MULTIPLIER
            * distance_moved_toward_food
            / MAX_DIST_FROM_FOOD
        )
        distance_to_other_snake_reward = (
            DISTANCE_TO_OTHER_SNAKE_MULTIPLIER
            * distance_moved_toward_other_snake
            / MAX_DIST_FROM_OTHER_SNAKE
        )

        reward += food_reward
        reward += distance_to_other_snake_reward

        reward = min(reward, 1.0)
        reward = max(reward, -1.0)
        return reward
