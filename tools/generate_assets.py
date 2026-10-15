"""Generate SVG sprite assets for the snake game.

All sprites are 64x64 with a transparent background:
  - fruit.svg: red circle, black border
  - wall_pipe_h.svg / wall_pipe_v.svg: yellow square, black border on the two
    non-joining sides only (h joins left+right, v joins top+bottom)
  - wall_corner_{up_right,right_down,down_left,left_up}.svg: yellow square
    with a rounded outer corner, black border on the two non-joining sides
  - snake1_*.svg / snake2_*.svg:
      head/tail: half-capsule (flat side joins the body), black border only
                 on the non-joining (rounded) edge
      body: full square, black border on the two non-joining sides only
                 (h joins left+right, v joins top+bottom)
      corner_{up_right,right_down,down_left,left_up}: square with rounded
                 outer corner, black border on the two non-joining sides

Directional and corner variants are drawn once in a base orientation and
rotated about the tile centre (32, 32).
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "resources"

BLACK = "#000000"
STROKE = 6
TILE = 64
CORNER_RADIUS = 16  # 1/4 of a tile

SNAKE_COLOURS = {1: "#00c000", 2: "#0000ff"}
WALL_COLOUR = "#ffff00"

DIRECTIONS = {"right": 0, "down": 90, "left": 180, "up": 270}
# Base corner/pipe piece joins top+right; rotating maps the other joins.
CORNER_JOININGS = {
    "up_right": 0,
    "right_down": 90,
    "down_left": 180,
    "left_up": 270,
}


def svg_wrapper(inner: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">\n'
        f"{inner.strip()}\n"
        "</svg>\n"
    )


def rotate(inner: str, degrees: int) -> str:
    if degrees == 0:
        return inner
    return f'<g transform="rotate({degrees} 32 32)">{inner}</g>'


# ---------------------------------------------------------------------------
# Static sprites
# ---------------------------------------------------------------------------

def fruit() -> str:
    r = TILE / 2 - STROKE / 2
    return svg_wrapper(
        f'<circle cx="32" cy="32" r="{r}" fill="#ff0000" '
        f'stroke="{BLACK}" stroke-width="{STROKE}"/>'
    )


# ---------------------------------------------------------------------------
# Snake / wall pieces (base orientation: joining top + right)
# ---------------------------------------------------------------------------

def half_capsule(colour: str) -> str:
    """Half-capsule pointing right: flat left edge (joins body), semicircular
    right edge. Border only on the non-joining edge."""
    fill = (
        f'<path d="M 4 4 L 32 4 A 28 28 0 0 1 32 60 L 4 60 Z" fill="{colour}"/>'
    )
    border = (
        '<path d="M 4 4 L 32 4 A 28 28 0 0 1 32 60 L 4 60" '
        f'fill="none" stroke="{BLACK}" stroke-width="{STROKE}" stroke-linecap="butt"/>'
    )
    return fill + "\n" + border


def pipe(colour: str) -> str:
    """Straight piece joining left+right: full square, black border on the two
    non-joining sides (top and bottom). Rotate 90 degrees for top+bottom.
    Used for both the wall pipes and the snake bodies."""
    fill = f'<rect x="0" y="0" width="{TILE}" height="{TILE}" fill="{colour}"/>'
    border = (
        f'<path d="M 0 0 L {TILE} 0 M 0 {TILE} L {TILE} {TILE}" '
        f'fill="none" stroke="{BLACK}" stroke-width="{STROKE}" stroke-linecap="butt"/>'
    )
    return fill + "\n" + border


def corner(colour: str) -> str:
    """Corner piece joining top+right: rounded outer (bottom-left) corner,
    black border on the two non-joining sides (left and bottom) only."""
    r = CORNER_RADIUS
    fill = (
        f'<path d="M 0 0 L {TILE} 0 L {TILE} {TILE} L {r} {TILE} '
        f'A {r} {r} 0 0 1 0 {TILE - r} Z" fill="{colour}"/>'
    )
    border = (
        f'<path d="M 0 0 L 0 {TILE - r} A {r} {r} 0 0 0 {r} {TILE} L {TILE} {TILE}" '
        f'fill="none" stroke="{BLACK}" stroke-width="{STROKE}" stroke-linecap="butt"/>'
    )
    return fill + "\n" + border


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {
        "fruit.svg": fruit(),
        "wall_pipe_h.svg": svg_wrapper(pipe(WALL_COLOUR)),
        "wall_pipe_v.svg": svg_wrapper(rotate(pipe(WALL_COLOUR), 90)),
    }
    for joining, deg in CORNER_JOININGS.items():
        files[f"wall_corner_{joining}.svg"] = svg_wrapper(
            rotate(corner(WALL_COLOUR), deg)
        )

    for n, colour in SNAKE_COLOURS.items():
        for facing, deg in DIRECTIONS.items():
            files[f"snake{n}_head_{facing}.svg"] = svg_wrapper(rotate(half_capsule(colour), deg))
            files[f"snake{n}_tail_{facing}.svg"] = svg_wrapper(rotate(half_capsule(colour), deg))
        for facing, deg in ("h", 0), ("v", 90):
            files[f"snake{n}_body_{facing}.svg"] = svg_wrapper(rotate(pipe(colour), deg))
        for joining, deg in CORNER_JOININGS.items():
            files[f"snake{n}_corner_{joining}.svg"] = svg_wrapper(
                rotate(corner(colour), deg)
            )

    for name, content in files.items():
        (OUT_DIR / name).write_text(content)
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
