from src.actors.base import PyGameObject

from src.game.screen import Screen
from src.config.enums import Colour, Direction

import pygame


class Snake(PyGameObject):
    body: list[tuple[int, int]] = []
    grow: bool = False
    body_colour: Colour = Colour.GREEN
    current_direction: Direction = Direction.RIGHT

    def __init__(
        self,
        screen: Screen,
        start_position: tuple[int, int],
        start_direction: Direction = Direction.RIGHT,
        body_colour: Colour = Colour.GREEN,
    ) -> None:
        super().__init__(screen)
        self.body: list[tuple[int, int]] = [start_position]
        self.current_direction = start_direction
        self.grow = False
        self.body_colour = body_colour

    def move(self) -> None:
        head_x, head_y = self.body[0]
        dx, dy = self.current_direction.value
        new_head = (head_x + dx, head_y + dy)
        self.body.insert(0, new_head)
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False

    def change_direction(self, new_direction: Direction) -> None:
        if (
            new_direction.value[0] * -1,
            new_direction.value[1] * -1,
        ) != self.current_direction:
            self.current_direction = new_direction

    def check_collision(self) -> bool:
        head = self.body[0]
        return (
            head in self.body[1:]
            or head[0] < 0
            or head[0] >= self.screen.n_cols
            or head[1] < 0
            or head[1] >= self.screen.n_rows
        )

    def draw(self) -> None:
        for segment in self.body:
            pygame.draw.rect(
                self.screen.surface,
                self.body_colour.value,
                (
                    segment[0] * self.screen.block_size,
                    segment[1] * self.screen.block_size,
                    self.screen.block_size,
                    self.screen.block_size,
                ),
            )
