from dataclasses import dataclass
from enum import Enum, unique
import random
import pygame
import torch
import numpy as np

from src.actors import Snake, Food
from src.game.screen import Screen
from src.config.enums import Colour, Direction
from src.ai.models import DQN

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
    previous_distance_to_food: float = 0.0
    new_distance_to_food: float = 0.0
    previous_distance_to_wall: float = 0.0
    new_distance_to_wall: float = 0.0
    is_food_eaten: bool = False
    is_game_over: bool = False


@unique
class StateMember(Enum):
    EDGE = 0
    EMPTY = 1
    SNAKE1 = 2
    SNAKE1_HEAD = 3
    SNAKE2 = 4
    SNAKE2_HEAD = 5
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
    ai_model: DQN | None = None
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
            _board_state[_pos[0] + 1, _pos[1] + 1] = StateMember.SNAKE1
        for _pos in self.snakes[1].body:
            _board_state[_pos[0] + 1, _pos[1] + 1] = StateMember.SNAKE2

        _board_state[
            self.snakes[0].body[0][0] + 1, self.snakes[0].body[0][1] + 1
        ] = StateMember.SNAKE1_HEAD
        _board_state[
            self.snakes[1].body[0][0] + 1, self.snakes[1].body[0][1] + 1
        ] = StateMember.SNAKE2_HEAD

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
        _ai_model = DQN(len(self.state[1]), 4).to(device)
        _ai_model.load_state_dict(torch.load("models/base_policy_net.pth"))
        self.ai_model = _ai_model

    def model_ai_move(self) -> Direction:
        self.current_ai_snake = 1
        state = self.training_ai_get_current_snake_state()

        state = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        if self.ai_model is not None:
            prediction = self.ai_model(state).max(1).indices.view(1, 1)
            return list(Direction)[prediction]
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

        for _snake_idx, _snake in enumerate(self.snakes):
            is_food_eaten = False
            is_game_over = False
            previous_direction = _snake.current_direction

            previous_distance_to_food = self.distance_to_food(_snake)
            previous_distance_to_wall = self.distance_to_wall(_snake)

            if _snake.is_ai:
                _snake.change_direction(self.ai_move())
                if self.is_ai_enabled:
                    _snake.move()
            else:
                _snake.change_direction(self.last_input_direction)
                _snake.move()

            new_distance_to_food = self.distance_to_food(_snake)
            new_distance_to_wall = self.distance_to_wall(_snake)

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
                case StateMember.SNAKE1:
                    colour = Colour.GREEN.value
                case StateMember.SNAKE2:
                    colour = Colour.BLUE.value
                case StateMember.FOOD:
                    colour = Colour.RED.value
                case StateMember.EDGE:
                    colour = Colour.YELLOW.value
                case StateMember.SNAKE1_HEAD:
                    colour = Colour.PURPLE.value
                case StateMember.SNAKE2_HEAD:
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

        start_pos_y += self.screen.block_size
        text = "Snake Pos X: {:.3f}".format(_snake_ai_state[0])
        self.screen.surface.blit(
            self.screen.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

        start_pos_y += self.screen.block_size
        text = "Snake Pos Y: {:.3f}".format(_snake_ai_state[1])
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

    def training_ai_get_current_snake_state(self) -> list[float]:
        state = []
        _snake = self.snakes[self.current_ai_snake]

        normalised_position_x = _snake.body[0][0] / self.screen.n_cols
        normalised_position_y = _snake.body[0][1] / self.screen.n_rows
        normalised_fruit_position_x = (
            self.food.position[0] / self.screen.n_cols
        )
        normalised_fruit_position_y = (
            self.food.position[1] / self.screen.n_cols
        )
        state.append(normalised_position_x)
        state.append(normalised_position_y)
        state.append(normalised_fruit_position_x)
        state.append(normalised_fruit_position_y)

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
        previous_direction = _snake.current_direction

        previous_distance_to_food = self.distance_to_food(_snake)
        previous_distance_to_wall = self.distance_to_wall(_snake)

        if self.current_ai_snake != 0:
            _snake.change_direction(input_direction)
            _snake.move()

        new_distance_to_food = self.distance_to_food(_snake)
        new_distance_to_wall = self.distance_to_wall(_snake)

        if _snake.body[0] == self.food.position:
            is_food_eaten = True
            _snake.eaten_fruit = True
            self.food.move(self.free_positions)

        if _snake.check_collision(self.filled_positions):
            is_game_over = True
            self.quit()

        if self.current_ai_snake == 0:
            self.current_ai_snake = (self.current_ai_snake + 1) % 2
            return

        rewardContext = RewardContext(
            previous_direction=previous_direction,
            new_direction=input_direction,
            previous_distance_to_food=previous_distance_to_food,
            new_distance_to_food=new_distance_to_food,
            previous_distance_to_wall=previous_distance_to_wall,
            new_distance_to_wall=new_distance_to_wall,
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
        distance_x = min(snake_x, self.screen.n_cols - snake_x)
        distance_y = min(snake_y + 1, self.screen.n_rows - snake_y)

        return min(distance_x, distance_y)

    def calculate_ai_reward(self, rewardContext: RewardContext) -> float:
        reward = 0.0
        if rewardContext.is_game_over:
            return -1.0
        if rewardContext.is_food_eaten:
            return 1.0
        if rewardContext.new_distance_to_wall == 1:
            return -1.0

        if (
            rewardContext.previous_direction.value[0]
            == rewardContext.new_direction.value[0] * -1
        ) and (
            rewardContext.previous_direction.value[1]
            == rewardContext.new_direction.value[1] * -1
        ):
            return -1.0
        distance_moved_toward_food = (
            rewardContext.previous_distance_to_food
            - rewardContext.new_distance_to_food
        )
        # distance_moved_toward_wall = (
        #     rewardContext.previous_distance_to_wall
        #     - rewardContext.new_distance_to_wall
        # )

        food_reward = 0.0
        wall_reward = 0.0

        # DEPENDANT ON MANHATTEN DISTANCE
        # TODO - set all this in training config
        MAX_DIST_FROM_FOOD = self.screen.n_cols + self.screen.n_rows
        MAX_DIST_FROM_WALL = self.screen.n_cols // 2
        FOOD_REWARD_MULTIPLIER = 10.0
        WALL_REWARD_MULTIPLIER = 1.0

        food_reward = (
            FOOD_REWARD_MULTIPLIER
            * distance_moved_toward_food
            / MAX_DIST_FROM_FOOD
        )
        # wall_reward = -(
        #     WALL_REWARD_MULTIPLIER
        #     * distance_moved_toward_wall
        #     / MAX_DIST_FROM_WALL
        # )

        reward += food_reward + wall_reward

        reward = min(reward, 1.0)
        reward = max(reward, -1.0)
        return reward
