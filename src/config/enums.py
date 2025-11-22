from enum import Enum, unique


@unique
class Colour(Enum):
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    GREY = (120, 120, 120)
    YELLOW = (255, 255, 0)
    PURPLE = (255, 0, 255)


@unique
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
