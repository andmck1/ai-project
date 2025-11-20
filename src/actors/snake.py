from src.actors.base import PyGameObject

from src.game.screen import Screen
from src.config.enums import Colour, Direction

import pygame


class Snake(PyGameObject):
    body: list[tuple[int, int]] = []
    previous_head_position = tuple[int, int]
    eaten_fruit: bool = False
    body_colour: Colour = Colour.GREEN
    current_direction: Direction = Direction.RIGHT
    is_ai: bool = False
    name: str = "NO NAME ASSIGNED"

    def __init__(
        self,
        screen: Screen,
        start_position: tuple[int, int],
        name: str,
        is_ai: bool = False,
        start_direction: Direction = Direction.RIGHT,
        body_colour: Colour = Colour.GREEN,
    ) -> None:
        super().__init__(screen)
        self.body: list[tuple[int, int]] = [start_position]
        self.current_direction = start_direction
        self.eaten_fruit = False
        self.body_colour = body_colour
        self.is_ai = is_ai
        self.name = name
        self.previous_head_position = start_position

    def move(self) -> None:
        head_position = self.body[0]
        self.previous_head_position = head_position
        current_direction = self.current_direction.value

        # TODO - Could make array instead and use numpy maths
        next_head_position = (
            head_position[0] + current_direction[0],
            head_position[1] + current_direction[1],
        )
        self.body.insert(0, next_head_position)
        if not self.eaten_fruit:
            self.body.pop()
        else:
            self.eaten_fruit = False

    def change_direction(self, new_direction: Direction) -> None:
        # Check not moving into self when length greater than 1
        if len(self.body) > 1:
            if (
                new_direction.value[0] * -1,
                new_direction.value[1] * -1,
            ) == self.current_direction.value:
                return None
        self.current_direction = new_direction

    def check_collision(self, filled_positions: list[tuple[int, int]]) -> bool:
        head = self.body[0]

        # If head in filled positions twice then occupying filled space
        if len([_pos for _pos in filled_positions if _pos == head]) > 1:
            return True

        # Check if OOB
        return (
            head[0] < 0
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
