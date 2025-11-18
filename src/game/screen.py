from dataclasses import dataclass

import pygame


@dataclass
class Screen:
    surface: pygame.surface.Surface
    width: int = 400
    height: int = 400
    block_size: int = 20

    def __init__(
        self,
        screen_width: int = 400,
        screen_height: int = 400,
        block_size: int = 20,
    ):
        self.width = screen_width
        self.height = screen_height
        self.block_size = block_size
        self.surface = pygame.display.set_mode((self.width, self.height))

    @property
    def n_rows(self) -> int:
        return self.height // self.block_size

    @property
    def n_cols(self) -> int:
        return self.width // self.block_size
