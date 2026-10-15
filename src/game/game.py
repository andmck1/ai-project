from src.config.pygame_import import pygame
from src.actors import Snake, Food
from src.game.state import GameState
from src.game.screen import Screen
from src.config.enums import Colour, Direction, StateMember
from src.config.datamodels import RewardContext
from src.ai.data import create_input_row

import random
from typing import Any
import numpy as np
import pandas as pd
from datetime import datetime as dt


class Game:
    state: GameState
    clock: pygame.time.Clock
    framerate: int = 8
    is_two_ai_playing: bool = False
    is_draw_enabled: bool = True
    is_debug_enabled: bool = True
    is_ai_enabled: bool = True

    def __init__(
        self,
        screen: Screen,
        window_title: str,
        framerate: int = 8,
        ai_mlflow_model_path_or_uri: str = "",
        is_two_ai_playing: bool = False,
        is_draw_enabled: bool = True,
        is_ai_random: bool = True,
        is_debug_enabled: bool = True,
        is_ai_enabled: bool = True,
        is_escape_pressed: bool = False,
    ) -> None:
        # properties
        self.is_two_ai_playing = is_two_ai_playing
        self.is_draw_enabled = is_draw_enabled
        self.framerate = framerate
        self.is_ai_random = is_ai_random
        self.is_debug_enabled = is_debug_enabled
        self.is_ai_enabled = is_ai_enabled
        self.clock = pygame.time.Clock()
        self.is_escape_pressed = False

        # state
        snakes = []
        food = Food(screen=screen)
        start_position_1 = (5, 5)
        start_position_2 = (screen.n_rows - 5, screen.n_cols - 5)
        if is_two_ai_playing:
            snakes = [
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
            snakes = [
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
        self.state = GameState(
            screen=screen,
            snakes=snakes,
            food=food,
            ai_model_mlflow_path_or_uri=ai_mlflow_model_path_or_uri,
        )
        if is_ai_enabled:
            if not is_ai_random:
                self.state.load_ai_model()
            if self.is_two_ai_playing:
                self.state.current_ai_snake = 0
            else:
                self.state.current_ai_snake = 1

        # init
        pygame.init()
        pygame.display.set_caption(window_title)
        self.state.food.move(self.state.filled_positions)

    def quit(self) -> None:
        pygame.quit()

    def handle_events(self) -> bool:
        # Hardcode for single player
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_escape_pressed = True
                self.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_escape_pressed = True
                    self.quit()
                    return False
                if event.key == pygame.K_UP:
                    self.state.last_input_direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.state.last_input_direction = Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    self.state.last_input_direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.state.last_input_direction = Direction.RIGHT

        return True

    def random_ai_move(self) -> Direction:
        """
        Choose a random move from list of all directions
        """
        random_direction: Direction = list(Direction)[
            random.randint(0, len(Direction) - 1)
        ]
        return random_direction

    def model_ai_move(self) -> Direction:
        """
        Get move for the AI by getting the current state for model input,
        creating a list of rows for each input direction, running inference
        and filtering for moves that would guarantee a loss (eg. going into
        a wall or another snake).
        """
        if self.state.ai_model is not None:
            start_time = dt.now()
            state = self.training_ai_get_current_snake_state()
            input_rows = []
            for direction in list(Direction):
                input_row = create_input_row(
                    self.state.current_ai_snake, state, 0, direction
                )
                input_rows.append(input_row)
            input_rows = pd.DataFrame(input_rows).drop("reward", axis=1)
            allowed_moves = self.check_allowed_moves(
                self.state.snakes[self.state.current_ai_snake]
            )
            directions = list(
                np.array(list(Direction))[
                    self.state.ai_model.predict(input_rows).argsort()[::-1]
                ]
            )
            direction = [
                direction
                for direction in directions
                if direction in allowed_moves
            ]
            inference_time = (dt.now() - start_time).total_seconds()
            self.state.ai_model_inference_times.append(inference_time)
            self.state.ai_model_inference_times = (
                self.state.ai_model_inference_times[1:]
            )
            if len(direction) > 0:
                return direction[0]
            else:
                # dead
                return Direction.UP
        else:
            print("Error, unbound AI")
            return Direction.UP

    def ai_move(self) -> Direction:
        if self.is_ai_random:
            return self.random_ai_move()
        else:
            return self.model_ai_move()

    def frame(self) -> bool:
        """
        Core game loop logic.
        """
        # TODO - could be more efficient to collect the rectangles
        # and display using .update() for draw calls

        if not self.handle_events():
            return False

        rewardContexts = []

        # loop over all snakes
        for _snake in self.state.snakes:
            # collect state
            is_food_eaten = False
            is_game_over = False
            previous_direction = _snake.current_direction
            _other_snake = self.state.snakes[
                (self.state.current_ai_snake + 1) % 2
            ]

            previous_distance_to_food = self.state.distance_to_food(_snake)
            previous_distance_to_wall = self.state.distance_to_wall(_snake)
            previous_distance_to_other_snake = (
                self.state.distance_to_closest_snake_segment(
                    _snake, _other_snake
                )
            )

            # if snake is AI controlled then move snake with AI generated move
            # otherwise take input from player
            if _snake.is_ai:
                if self.is_ai_enabled:
                    _snake.change_direction(self.ai_move())
                    _snake.move()
                    if self.is_two_ai_playing:
                        self.state.increment_current_ai_index()
            else:
                _snake.change_direction(self.state.last_input_direction)
                _snake.move()

            # update new state based on previous move
            new_distance_to_food = self.state.distance_to_food(_snake)
            new_distance_to_wall = self.state.distance_to_wall(_snake)
            new_distance_to_other_snake = (
                self.state.distance_to_closest_snake_segment(
                    _snake, _other_snake
                )
            )

            # check if food eaten
            if _snake.body[0] == self.state.food.position:
                is_food_eaten = True
                _snake.eaten_fruit = True
                self.state.food.move(self.state.free_positions)

            # check if collision
            if _snake.check_collision(self.state.filled_positions):
                is_game_over = True
                print(_snake.name + " lost")
                self.quit()

            # create reward (used for training and tracking state)
            rewardContext = RewardContext(
                previous_direction=previous_direction,
                new_direction=self.state.last_input_direction,
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

            # exit event if game over
            if is_game_over:
                return False

        # draw screen
        if self.is_draw_enabled:
            self.state.screen.surface.fill(Colour.BLACK.value)
            self.state.screen.draw_border()
            for _snake in self.state.snakes:
                _snake.draw()
            self.state.food.draw()
            if self.is_debug_enabled:
                self.draw_debug(rewardContexts)
            pygame.display.flip()

        # move time
        self.clock.tick(self.framerate)

        return True

    def draw_debug(self, rewardContexts: list[RewardContext]) -> None:
        """
        Draws debug information onto game screen.
        """
        debug_start_x = self.state.screen.debug_start_x
        assert debug_start_x is not None

        # draw ai snake state
        current_ai_snake = self.state.current_ai_snake
        start_pos_x = debug_start_x
        start_pos_y = 0
        for current_snake_index in range(2):
            start_pos_x = debug_start_x
            self.state.screen.draw_text(
                f"Snake State {current_snake_index}", start_pos_x, start_pos_y
            )
            self.state.current_ai_snake = current_snake_index
            _snake_ai_state = self.training_ai_get_current_snake_state()

            texts = [
                "Distance to Wall X: {:.3f}".format(_snake_ai_state[0]),
                "Distance to Wall Y: {:.3f}".format(_snake_ai_state[1]),
                "Fruit Pos X: {:.3f}".format(_snake_ai_state[2]),
                "Fruit Pos Y: {:.3f}".format(_snake_ai_state[3]),
                "Closest Segment Pos X: {:.3f}".format(_snake_ai_state[4]),
                "Closest Segment Pos Y: {:.3f}".format(_snake_ai_state[5]),
                "State Relative to Head (+1, 0): {}".format(
                    _snake_ai_state[6].name
                ),
                "State Relative to Head (-1, 0): {}".format(
                    _snake_ai_state[7].name
                ),
                "State Relative to Head (0, +1): {}".format(
                    _snake_ai_state[8].name
                ),
                "State Relative to Head (0, -1): {}".format(
                    _snake_ai_state[9].name
                ),
                "Current Direction: {}".format(_snake_ai_state[10].name),
            ]

            for text in texts:
                start_pos_y += self.state.screen.block_size
                self.state.screen.draw_text(text, start_pos_x, start_pos_y)

            # draw rewards
            for i in range(2):
                start_pos_x = 0
                start_pos_y_reward = (
                    self.state.screen.block_size * self.state.screen.n_rows
                    + i * self.state.screen.block_size
                    + self.state.screen.edge_size
                    * self.state.screen.block_size
                    * 2
                )
                reward = self.calculate_ai_reward(rewardContexts[i])
                text = f"Snake {i} Reward: {reward:.3f}"
                self.state.screen.draw_text(
                    text, start_pos_x, start_pos_y_reward
                )

            start_pos_y += self.state.screen.block_size * 3

        self.state.current_ai_snake = current_ai_snake

    def training_ai_frame(self) -> None:
        self.state.screen.surface.fill(Colour.BLACK.value)

    def training_ai_end_frame(self) -> None:
        if self.is_draw_enabled:
            self.state.food.draw()
            for _snake in self.state.snakes:
                _snake.draw()
            pygame.display.flip()

        self.clock.tick(self.framerate)

    def training_ai_get_current_snake_state(self) -> list[Any]:
        """
        Generate state for AI snake. Used primarily for training but also 
        useful for inference, where input state is required to create input 
        rows for model.
        """
        state = []
        _snake = self.state.snakes[self.state.current_ai_snake]
        _other_snake = self.state.snakes[(self.state.current_ai_snake + 1) % 2]
        _board_state, _ = self.state.board_state
        
        _distances_from_head_to_other_snake = (
            self.state.distances_from_head_to_other_snake(_snake, _other_snake)
        )
        _closest_other_snake_segment = self.state.closest_other_snake_segment(
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
            min(
                _snake.body[0][0],
                abs(_snake.body[0][0] - self.state.screen.n_cols),
            )
            / self.state.screen.n_cols
        )
        normalised_distance_to_wall_y = (
            min(
                _snake.body[0][1],
                abs(_snake.body[0][1] - self.state.screen.n_rows),
            )
            / self.state.screen.n_rows
        )

        normalised_fruit_position_x_delta = (
            _snake.body[0][0] - self.state.food.position[0]
        ) / self.state.screen.n_cols
        normalised_fruit_position_y_delta = (
            _snake.body[0][1] - self.state.food.position[1]
        ) / self.state.screen.n_cols

        _closest_segment_x_delta = (
            _distance_to_closest_snake_segment[0] / self.state.screen.n_cols
        )
        _closest_segment_y_delta = (
            _distance_to_closest_snake_segment[1] / self.state.screen.n_rows
        )

        state.append(normalised_distance_to_wall_x)
        state.append(normalised_distance_to_wall_y)
        state.append(normalised_fruit_position_x_delta)
        state.append(normalised_fruit_position_y_delta)
        state.append(_closest_segment_x_delta)
        state.append(_closest_segment_y_delta)
        state += _statemembers_around_head
        state.append(_snake.current_direction)

        return state

    def check_allowed_moves(self, snake: Snake) -> list[Direction]:
        """
        Checks for any game ending moves. If a given direction isn't empty
        and isn't food then it should not be used.
        """
        current_board_state, _ = self.state.board_state
        allowed_directions = []
        for direction in list(Direction):
            check_pos = (
                snake.body[0][0] + direction.value[0] + 1,
                snake.body[0][1] + direction.value[1] + 1,
            )
            check_state = current_board_state[check_pos]
            if check_state in [StateMember.EMPTY, StateMember.FOOD]:
                allowed_directions.append(direction)
        return allowed_directions

    def training_ai_step(
        self, input_direction: Direction
    ) -> tuple[list[int] | None, float, bool] | None | int:
        """
        Specific code for training the AI by having the AI play against itself
        by controlling both snakes.
        """

        # init
        reward = 0.0
        next_state = []
        is_game_over = False
        is_food_eaten = False
        
        # get or generate relevant state
        _snake = self.state.snakes[self.state.current_ai_snake]
        allowed_directions = self.check_allowed_moves(_snake)
        if input_direction not in allowed_directions:
            return -1
        _other_snake = self.state.snakes[(self.state.current_ai_snake + 1) % 2]

        previous_direction = _snake.current_direction

        previous_distance_to_food = self.state.distance_to_food(_snake)
        previous_distance_to_wall = self.state.distance_to_wall(_snake)
        previous_distance_to_other_snake = (
            self.state.distance_to_closest_snake_segment(_snake, _other_snake)
        )
        
        # move snake
        _snake.change_direction(input_direction)
        _snake.move()
        
        # get updated state
        new_distance_to_food = self.state.distance_to_food(_snake)
        new_distance_to_wall = self.state.distance_to_wall(_snake)
        new_distance_to_other_snake = (
            self.state.distance_to_closest_snake_segment(_snake, _other_snake)
        )
        
        # check for food eaten
        if _snake.body[0] == self.state.food.position:
            is_food_eaten = True
            _snake.eaten_fruit = True
            self.state.food.move(self.state.free_positions)
        
        # check for game over state
        if _snake.check_collision(self.state.filled_positions):
            is_game_over = True
            self.quit()
        
        # get reward update
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
        
        # update state and increment AI counter
        next_state = self.training_ai_get_current_snake_state()
        self.state.increment_current_ai_index()
        return next_state, reward, is_game_over

    def calculate_ai_reward(self, rewardContext: RewardContext) -> float:
        """
        This is the core function for calculating reward.

        If it was game over or next to wall or head of other snake
        then return max negative reward.

        If food just eaten return maximum reward.

        If direction provided tried to move the snake backward, return maximum
        negative reward.

        Otherwise, calculate reward as a function of distance to food
        and distance to wall, where moving toward food increases reward
        and moving away reduces reward.
        """
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
        MAX_DIST_FROM_FOOD = (
            self.state.screen.n_cols + self.state.screen.n_rows
        )
        MAX_DIST_FROM_OTHER_SNAKE = (
            self.state.screen.n_cols + self.state.screen.n_rows
        )
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
