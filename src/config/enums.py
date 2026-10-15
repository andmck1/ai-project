from enum import Enum, unique


@unique
class Colour(Enum):
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    GREY = (120, 120, 120)
    DARK_GREY = (45, 45, 45)
    YELLOW = (255, 255, 0)
    PURPLE = (255, 0, 255)


@unique
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


@unique
class StateMember(Enum):
    EDGE = 0
    EMPTY = 1
    SNAKE0 = 2
    SNAKE0_HEAD = 3
    SNAKE1 = 4
    SNAKE1_HEAD = 5
    FOOD = 6
