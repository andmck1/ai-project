from src.actors.base import PyGameObject

from src.game.screen import Screen
from src.config.enums import Colour

import random
import pygame


class Food(PyGameObject):
    colour: Colour
    position: tuple[int, int]

    def __init__(self, screen: Screen, colour: Colour = Colour.RED) -> None:
        super().__init__(screen)
        self.colour = colour
        self.position = (0, 0)

    def move(self, free_positions: list[tuple[int, int]]) -> None:
        self.position = free_positions[
            random.randint(0, len(free_positions) - 1)
        ]

    def draw(self) -> None:
        pygame.draw.rect(
            self.screen.surface,
            self.colour.value,
            (
                (self.position[0] + 1) * self.screen.block_size,
                (self.position[1] + 1) * self.screen.block_size,
                self.screen.block_size,
                self.screen.block_size,
            ),
        )
