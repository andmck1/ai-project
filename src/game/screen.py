from dataclasses import dataclass

from src.config.enums import Colour
from src.config.pygame_import import pygame


@dataclass
class Screen:
    surface: pygame.surface.Surface
    font: pygame.font.Font
    all_positions: list[tuple[int, int]]
    total_width: int
    total_height: int
    debug_start_x: int
    debug_start_y: int
    board_width: int = 400
    board_height: int = 400
    edge_size: int = 1
    block_size: int = 20
    border_colour: Colour = Colour.YELLOW

    def __init__(
        self,
        board_width: int = 400,
        board_height: int = 400,
        block_size: int = 20,
        edge_size: int = 1,
        is_draw_enabled: bool = True,
        border_colour: Colour = Colour.YELLOW,
    ) -> None:
        self.board_width = board_width + block_size * edge_size * 2
        self.board_height = board_height + block_size * edge_size * 2
        self.block_size = block_size
        self.edge_size = edge_size
        self.border_colour = border_colour

        debug_margin = block_size * 4
        debug_height = block_size * 4
        debug_width = self.board_width + debug_margin
        debug_separation = debug_margin
        self.debug_start_x = self.board_width + debug_separation
        self.debug_start_y = self.board_height

        flags = 0
        if not is_draw_enabled:
            flags = pygame.HIDDEN

        pygame.font.init()
        self.font = pygame.font.Font(None, 30)

        self.total_height = self.board_height + debug_separation + debug_height
        self.total_width = self.board_width + debug_separation + debug_width
        self.surface = pygame.display.set_mode(
            (self.total_width, self.total_height),
            flags=flags,
        )

        self.all_positions = [
            (i, j) for i in range(self.n_cols) for j in range(self.n_rows)
        ]

    def draw_text(self, text: str, start_pos_x: int, start_pos_y: int) -> None:
        self.surface.blit(
            self.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

    def draw_border(self) -> None:
        for i in range(0, self.n_rows + 2):
            for j in range(0, self.n_cols + 2):
                if (
                    (i != 0)
                    and (j != 0)
                    and (i <= self.n_rows)
                    and (j <= self.n_cols)
                ):
                    continue
                else:
                    pygame.draw.rect(
                        self.surface,
                        self.border_colour.value,
                        (
                            i * self.block_size,
                            j * self.block_size,
                            self.block_size,
                            self.block_size,
                        ),
                    )

    @property
    def n_rows(self) -> int:
        return (self.board_height - self.edge_size * 2) // self.block_size

    @property
    def n_cols(self) -> int:
        return (self.board_width - self.edge_size * 2) // self.block_size
