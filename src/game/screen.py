from dataclasses import dataclass
import pygame


@dataclass
class Screen:
    surface: pygame.surface.Surface
    font: pygame.font.Font
    all_positions: list[tuple[int, int]]
    width: int = 400
    height: int = 400
    block_size: int = 20
    debug_width: int = 0
    debug_height: int = 0
    debug_separation: int = 0

    def __init__(
        self,
        screen_width: int = 400,
        screen_height: int = 400,
        block_size: int = 20,
        is_draw_enabled: bool = True,
    ) -> None:
        self.width = screen_width
        self.height = screen_height
        self.block_size = block_size

        self.debug_height = self.height + block_size * 10
        self.debug_width = self.width + block_size * 10
        self.debug_separation = block_size * 2

        flags = 0
        if not is_draw_enabled:
            flags = pygame.HIDDEN

        pygame.font.init()
        self.font = pygame.font.Font(None, 30)

        self.surface = pygame.display.set_mode(
            (
                self.width + self.debug_width + self.debug_separation,
                self.height + self.debug_height,
            ),
            flags=flags,
        )

        self.all_positions = [
            (i, j) for i in range(self.n_cols) for j in range(self.n_rows)
        ]

    @property
    def n_rows(self) -> int:
        return self.height // self.block_size

    @property
    def n_cols(self) -> int:
        return self.width // self.block_size
