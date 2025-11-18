from abc import ABC, abstractmethod

from src.game.screen import Screen


class PyGameObject(ABC):
    screen: Screen

    def __init__(self, screen: Screen) -> None:
        self.screen = screen

    @abstractmethod
    def draw(self) -> None:
        pass
