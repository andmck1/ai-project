from dataclasses import dataclass

from src.config.enums import Colour
from src.config.pygame_import import pygame
from src.game.sprites import load_sprites


@dataclass
class Screen:
    surface: pygame.surface.Surface
    font: pygame.font.Font
    all_positions: list[tuple[int, int]]
    total_width: int
    total_height: int
    debug_start_x: int | None
    debug_start_y: int | None
    sprites: dict[str, pygame.surface.Surface]
    board_width: int = 800
    board_height: int = 800
    edge_size: int = 1
    block_size: int = 40
    border_colour: Colour = Colour.YELLOW
    is_debug_enabled: bool = False

    def __init__(
        self,
        board_width: int = 40*20,
        board_height: int = 40*20,
        block_size: int = 40,
        edge_size: int = 1,
        is_draw_enabled: bool = True,
        is_debug_enabled: bool = False,
        border_colour: Colour = Colour.YELLOW,
    ) -> None:
        self.board_width = board_width + block_size * edge_size * 2
        self.board_height = board_height + block_size * edge_size * 2
        self.block_size = block_size
        self.edge_size = edge_size
        self.border_colour = border_colour
        self.is_debug_enabled = is_debug_enabled

        if is_debug_enabled:
            debug_margin = block_size * 4
            debug_height = block_size * 4
            debug_width = self.board_width + debug_margin
            debug_separation = debug_margin
            self.debug_start_x = self.board_width + debug_separation
            self.debug_start_y = self.board_height
        else:
            debug_margin = 0
            debug_height = 0
            debug_width = 0
            debug_separation = 0
            self.debug_start_x = None
            self.debug_start_y = None

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

        self.sprites = load_sprites(self.block_size)

        self.all_positions = [
            (i, j) for i in range(self.n_cols) for j in range(self.n_rows)
        ]

    def draw_text(self, text: str, start_pos_x: int, start_pos_y: int) -> None:
        self.surface.blit(
            self.font.render(text, True, (255, 255, 255)),
            (start_pos_x, start_pos_y),
        )

    def blit_sprite(self, name: str, grid_position: tuple[int, int]) -> None:
        """Blit a named sprite onto a grid cell (0-indexed, board space)."""
        surface = self.sprites.get(name)
        if surface is None:
            return
        self.surface.blit(
            surface,
            (
                grid_position[0] * self.block_size,
                grid_position[1] * self.block_size,
            ),
        )

    def draw_border(self) -> None:
        # corners (name describes which two sides the piece joins)
        self.blit_sprite("wall_corner_right_down", (0, 0))
        self.blit_sprite("wall_corner_down_left", (self.n_cols + 1, 0))
        self.blit_sprite("wall_corner_up_right", (0, self.n_rows + 1))
        self.blit_sprite("wall_corner_left_up", (self.n_cols + 1, self.n_rows + 1))
        # straight edges
        for col in range(1, self.n_cols + 1):
            self.blit_sprite("wall_pipe_h", (col, 0))
            self.blit_sprite("wall_pipe_h", (col, self.n_rows + 1))
        for row in range(1, self.n_rows + 1):
            self.blit_sprite("wall_pipe_v", (0, row))
            self.blit_sprite("wall_pipe_v", (self.n_cols + 1, row))

    @property
    def n_rows(self) -> int:
        return (
            self.board_height - self.edge_size * 2 * self.block_size
        ) // self.block_size

    @property
    def n_cols(self) -> int:
        return (
            self.board_width - self.edge_size * 2 * self.block_size
        ) // self.block_size
