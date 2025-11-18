from src.actors import Snake, Food
from src.game.screen import Screen
from src.config.enums import Colour, Direction

import pygame


class Game:
    screen: Screen
    snake: Snake
    food: Food
    clock: pygame.time.Clock
    tick_length: int = 8

    def __init__(
        self, screen: Screen, window_title: str, tick_length: int = 8
    ) -> None:
        self.screen = screen
        self.snake = Snake(screen=screen, start_position=(5, 5))
        self.food = Food(screen=screen)
        self.clock = pygame.time.Clock()
        self.tick_length = tick_length

        pygame.init()
        pygame.display.set_caption(window_title)

    def quit(self):
        pygame.quit()
        return False

    def frame(self) -> bool:
        self.screen.surface.fill(Colour.BLACK.value)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit()
                if event.key == pygame.K_UP:
                    self.snake.change_direction(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    self.snake.change_direction(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    self.snake.change_direction(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.snake.change_direction(Direction.RIGHT)

        self.snake.move()

        if self.snake.body[0] == self.food.position:
            self.snake.grow = True
            # TODO - don't need new food, just to move food
            self.food = Food(self.screen)

        if self.snake.check_collision():
            print("DEAD")
            self.quit()

        self.snake.draw()
        self.food.draw()

        pygame.display.flip()
        self.clock.tick(self.tick_length)

        return True
