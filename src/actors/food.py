import random

from src.actors.base import PyGameObject
from src.config.enums import Colour
from src.game.screen import Screen


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
        # board coordinates are 0-indexed; the wall occupies grid index 0,
        # so offset by +1 to match the snake's draw position
        self.screen.blit_sprite(
            "fruit",
            (
                self.position[0] + 1,
                self.position[1] + 1,
            ),
        )
