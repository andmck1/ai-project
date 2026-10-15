from src.actors.base import PyGameObject
from src.config.enums import Colour, Direction
from src.game.screen import Screen


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

    @property
    def sprite_id(self) -> str:
        return "2" if self.body_colour == Colour.BLUE else "1"

    def draw(self) -> None:
        body_set = set(self.body)
        for index, segment in enumerate(self.body):
            sprite = self.screen.sprites.get(
                self._sprite_name(index, body_set)
            )
            if sprite is None:
                continue
            self.screen.surface.blit(
                sprite,
                (
                    (segment[0] + 1) * self.screen.block_size,
                    (segment[1] + 1) * self.screen.block_size,
                ),
            )

    def _sprite_name(
        self,
        index: int,
        body_set: set[tuple[int, int]],
    ) -> str:
        prefix = f"snake{self.sprite_id}"
        position = self.body[index]

        # head faces the current direction of travel
        if index == 0:
            return f"{prefix}_head_{self.current_direction.name.lower()}"

        # tail: rounded end points away from the rest of the body
        if index == len(self.body) - 1:
            previous = self.body[index - 1]
            if position[0] > previous[0]:
                facing = "right"
            elif position[0] < previous[0]:
                facing = "left"
            elif position[1] > previous[1]:
                facing = "down"
            else:
                facing = "up"
            return f"{prefix}_tail_{facing}"

        # body: pick the piece matching the two segments it actually links
        def direction_to(other: tuple[int, int]) -> str:
            dx = other[0] - position[0]
            dy = other[1] - position[1]
            if dx == 1:
                return "right"
            if dx == -1:
                return "left"
            if dy == 1:
                return "down"
            return "up"

        joins = {
            direction_to(self.body[index - 1]),
            direction_to(self.body[index + 1]),
        }

        if joins == {"left", "right"}:
            return f"{prefix}_body_h"
        if joins == {"up", "down"}:
            return f"{prefix}_body_v"
        corners = {
            frozenset({"up", "right"}): "corner_up_right",
            frozenset({"right", "down"}): "corner_right_down",
            frozenset({"down", "left"}): "corner_down_left",
            frozenset({"left", "up"}): "corner_left_up",
        }
        corner_name = corners.get(frozenset(joins))
        if corner_name is not None:
            return f"{prefix}_{corner_name}"

        # disconnected segment (shouldn't happen); fall back to a straight piece
        return f"{prefix}_body_h"
