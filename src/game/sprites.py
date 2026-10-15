"""Load SVG sprite assets as scaled pygame surfaces.

Sprites live in `src/resources` (see `tools/generate_assets.py`) and are
rasterised with cairosvg at startup, then scaled to the game's block size.
"""

import io
import sys
from pathlib import Path

import cairosvg

from src.config.pygame_import import pygame

if getattr(sys, "frozen", False):
    # PyInstaller layout: resources bundled into `_internal`
    RESOURCES_DIR = Path(sys.executable).parent / "_internal" / "resources"
else:
    RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"

# Render at high resolution for a clean downscale to the block size
_RENDER_SIZE = 256


def load_sprites(
    block_size: int,
) -> dict[str, pygame.surface.Surface]:
    """Load every SVG in the resources dir as a block_size x block_size
    surface, keyed by the SVG file name (without extension)."""
    sprites: dict[str, pygame.surface.Surface] = {}
    for svg_file in sorted(RESOURCES_DIR.glob("*.svg")):
        png_data = cairosvg.svg2png(
            url=str(svg_file),
            output_width=_RENDER_SIZE,
            output_height=_RENDER_SIZE,
        )
        surface = pygame.image.load(io.BytesIO(png_data), "png").convert_alpha()
        if surface.get_size() != (block_size, block_size):
            surface = pygame.transform.smoothscale(
                surface,
                (block_size, block_size),
            )
        sprites[svg_file.stem] = surface
    return sprites
