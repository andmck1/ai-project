from enum import Enum, unique
import random
import pygame
import torch

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


@unique
class StateMember(Enum):
    EMPTY = 0
    SNAKE1 = 1
    SNAKE2 = 2
    FOOD = 3


class Game:
    screen: Screen
    snakes: list[Snake]
    food: Food
    clock: pygame.time.Clock
    tick_length: int = 8
    all_positions: list[tuple[int, int]]
    is_two_ai_playing: bool = False
    last_input_direction: Direction = Direction.RIGHT
    is_draw_enabled: bool = True
    current_ai_snake: int = 0
    ai_model: DQN

    def __init__(
        self,
        screen: Screen,
        window_title: str,
        tick_length: int = 8,
        is_two_ai_playing: bool = False,
        is_draw_enabled: bool = True,
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
        self.tick_length = tick_length
        self.all_positions = [
            (i, j)
            for i in range(self.screen.n_cols)
            for j in range(self.screen.n_rows)
        ]
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
    def state(self) -> list[StateMember]:
        _board_state = {
            (i, j): StateMember.EMPTY
            for i in range(self.screen.n_rows)
            for j in range(self.screen.n_cols)
        }
        for _pos in self.snakes[0].body:
            _board_state[_pos] = StateMember.SNAKE1
        for _pos in self.snakes[1].body:
            _board_state[_pos] = StateMember.SNAKE2

        _board_state[self.food.position] = StateMember.FOOD
        _state = [
            _board_state[(i, j)]
            for i in range(self.screen.n_rows)
            for j in range(self.screen.n_cols)
        ]
        return _state

    def quit(self) -> bool:
        pygame.quit()
        return False

    def handle_events(self) -> bool:
        # Hardcode for single player
        # TODO - need to make it so either one and one AI
        # or two AI
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
        _ai_model = DQN(400, 4).to(device)
        _ai_model.load_state_dict(torch.load("models/base_policy_net.pth"))
        self.ai_model = _ai_model

    def model_ai_move(self) -> Direction:
        state = []
        _state = self.state
        for i in range(len(_state)):
            _statemember = _state[i]
            if _statemember == StateMember.EMPTY:
                state.append(0)
            elif _statemember == StateMember.FOOD:
                state.append(1)
            elif _statemember == StateMember.SNAKE2:
                state.append(2)
            else:
                state.append(3)

        state = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        prediction = self.ai_model(state).max(1).indices.view(1, 1)
        return list(Direction)[prediction]

    def frame(self) -> bool:
        self.screen.surface.fill(Colour.BLACK.value)

        if not self.handle_events():
            return False

        for _snake in self.snakes:
            if _snake.is_ai:
                # _snake.change_direction(self.random_ai_move())
                _snake.change_direction(self.model_ai_move())
            else:
                _snake.change_direction(self.last_input_direction)
            _snake.move()

            if _snake.body[0] == self.food.position:
                _snake.eaten_fruit = True
                self.food.move(self.free_positions)

            if _snake.check_collision(self.filled_positions):
                print(_snake.name + " lost")
                self.quit()
                return False

            if self.is_draw_enabled:
                _snake.draw()

        if self.is_draw_enabled:
            self.food.draw()
            # TODO - could be more efficient to collect the rectangles
            # and display using .update()
            pygame.display.flip()

        self.clock.tick(self.tick_length)

        return True

    def training_ai_frame(self) -> None:
        self.screen.surface.fill(Colour.BLACK.value)

    def training_ai_end_frame(self) -> None:
        self.clock.tick(self.tick_length)

    def training_ai_get_current_snake_state(self) -> list[int]:
        _current_snake_state_member = (
            StateMember.SNAKE1
            if self.current_ai_snake == 0
            else StateMember.SNAKE2
        )
        state = []

        _state = self.state
        for i in range(len(_state)):
            _statemember = _state[i]
            if _statemember == StateMember.EMPTY:
                state.append(0)
            elif _statemember == StateMember.FOOD:
                state.append(1)
            elif _statemember == _current_snake_state_member:
                state.append(2)
            else:
                state.append(3)

        return state

    # simulate env.step in gym
    # step the game by input direction and return:
    # previous state from current snake point of view,
    # new state from current snake point of view,
    # reward and if game ended
    def training_ai_step(
        self, input_direction: Direction
    ) -> tuple[list[int], float, bool]:
        reward = 0.0
        next_state = []
        game_over = False

        _snake = self.snakes[self.current_ai_snake]

        _snake.change_direction(input_direction)
        _snake.move()
        if _snake.body[0] == self.food.position:
            _snake.eaten_fruit = True
            self.food.move(self.free_positions)
            reward += 10.0

        if _snake.check_collision(self.filled_positions):
            game_over = True
            self.quit()
            reward = -1.0
        else:
            reward += 1.0

        next_state = self.training_ai_get_current_snake_state()

        self.current_ai_snake = (self.current_ai_snake + 1) % 2

        return next_state, reward, game_over
