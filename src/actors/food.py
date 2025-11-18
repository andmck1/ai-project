from src.actors.base import PyGameObject

from src.game.screen import Screen
from src.config.enums import Colour

import random
import pygame


class Food(PyGameObject):
    colour: Colour

    def __init__(self, screen: Screen, colour: Colour = Colour.RED) -> None:
        super().__init__(screen)
        self.colour = colour
        self.position = self.init_position()

    def init_position(self) -> tuple[int, int]:
        return (
            random.randint(0, self.screen.n_cols - 1),
            random.randint(0, self.screen.n_rows - 1),
        )

    def draw(self) -> None:
        pygame.draw.rect(
            self.screen.surface,
            self.colour.value,
            (
                self.position[0] * self.screen.block_size,
                self.position[1] * self.screen.block_size,
                self.screen.block_size,
                self.screen.block_size,
            ),
        )
