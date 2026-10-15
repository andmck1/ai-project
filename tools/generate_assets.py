"""Generate SVG sprite assets for the snake game.

All sprites are 64x64 with a transparent background:
  - fruit.svg: red circle, black border
  - wall_pipe_h.svg / wall_pipe_v.svg: yellow square, black border on the two
    non-joining sides only (h joins left+right, v joins top+bottom)
  - wall_corner_{up_right,right_down,down_left,left_up}.svg: yellow square
    with a rounded outer corner, black border on the two non-joining sides
  - snake1_*.svg / snake2_*.svg:
      head/tail: half-capsule (flat side joins the body), black border on
                 every edge except the flat joining edge, so the outline runs
                 continuously from the joining edge, round the semicircular
                 end, and back to the joining edge
      body: full square, black border on the two non-joining sides only
                 (h joins left+right, v joins top+bottom)
      corner_{up_right,right_down,down_left,left_up}: square with rounded
                 outer corner, black border on the two non-joining sides,
                 plus a black square on the inner corner of the bend (where
                 the two joining edges meet) so the outline stays continuous
                 with the neighbouring pieces

Borders are drawn as strokes whose centreline is inset by STROKE / 2 from the
sprite outline, so the outer edge of the border coincides exactly with the
outline and the full STROKE width is visible (nothing is clipped by the tile
edge, and nothing spills outside the rounded shapes).

Directional and corner variants are drawn once in a base orientation and
rotated about the tile centre (32, 32).
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "resources"

BLACK = "#000000"
STROKE = 12
HALF = STROKE / 2
TILE = 64
CORNER_RADIUS = 16  # 1/4 of a tile

SNAKE_COLOURS = {1: "#0ACF83", 2: "#1ABCFE"}
WALL_COLOUR = "#FFD02E"
FRUIT_COLOUR = "#F24E1E"

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


def stroke_path(d: str) -> str:
    return (
        f'<path d="{d}" fill="none" stroke="{BLACK}" '
        f'stroke-width="{STROKE}" stroke-linecap="butt" stroke-linejoin="miter"/>'
    )


# ---------------------------------------------------------------------------
# Static sprites
# ---------------------------------------------------------------------------


def fruit() -> str:
    r = TILE / 2 - HALF
    return svg_wrapper(
        f'<circle cx="32" cy="32" r="{r}" fill="{FRUIT_COLOUR}" '
        f'stroke="{BLACK}" stroke-width="{STROKE}"/>'
    )


# ---------------------------------------------------------------------------
# Snake / wall pieces (base orientation: joining top + right)
# ---------------------------------------------------------------------------


def half_capsule(colour: str) -> str:
    """Half-capsule pointing right: flat left edge at x=0 (joins body),
    straight top/bottom edges from x=0 to x=32, then a semicircular end of
    radius 32 centred on (32, 32).

    The border runs along the top edge, round the semicircle, and back along
    the bottom edge, leaving only the flat joining edge open. Its centreline
    is inset by HALF from the outline (arc radius 32 - HALF), so the ring
    spans radius 20..32 and the outer edge is exactly the sprite outline.
    """
    fill = f'<path d="M 0 0 L 32 0 A 32 32 0 0 1 32 64 L 0 64 Z" fill="{colour}"/>'
    r = 32 - HALF
    top = HALF
    bottom = TILE - HALF
    border = stroke_path(
        f"M 0 {top} L 32 {top} A {r} {r} 0 0 1 32 {bottom} L 0 {bottom}"
    )
    return fill + "\n" + border


def pipe(colour: str) -> str:
    """Straight piece joining left+right: full square, black border on the two
    non-joining sides (top and bottom). Rotate 90 degrees for top+bottom.
    Used for both the wall pipes and the snake bodies."""
    fill = (
        f'<rect x="0" y="0" width="{TILE}" height="{TILE}" fill="{colour}"/>'
    )
    top = HALF
    bottom = TILE - HALF
    border = stroke_path(
        f"M 0 {top} L {TILE} {top} M 0 {bottom} L {TILE} {bottom}"
    )
    return fill + "\n" + border


def corner(colour: str, inner_square: bool = False) -> str:
    """Corner piece joining top+right: rounded outer (bottom-left) corner of
    radius CORNER_RADIUS, black border on the two non-joining sides (left and
    bottom) only. The border centreline is inset by HALF from the outline, with
    a concentric arc of radius CORNER_RADIUS - HALF.

    If inner_square is True, a STROKE x STROKE black square is added at the
    top-right of the tile (the inner corner of the bend, opposite the rounded
    outer corner), joining the borders of the pieces above and to the right.
    """
    r = CORNER_RADIUS
    fill = (
        f'<path d="M 0 0 L {TILE} 0 L {TILE} {TILE} L {r} {TILE} '
        f'A {r} {r} 0 0 1 0 {TILE - r} Z" fill="{colour}"/>'
    )
    ir = r - HALF  # centreline arc radius
    left = HALF  # x of left border centreline
    bottom = TILE - HALF  # y of bottom border centreline
    cx, cy = r, TILE - r  # arc centre
    border = stroke_path(
        f"M {left} 0 L {left} {cy} "
        f"A {ir} {ir} 0 0 0 {cx} {bottom} "
        f"L {TILE} {bottom}"
    )
    parts = [fill, border]
    if inner_square:
        parts.append(
            f'<rect x="{TILE - STROKE}" y="0" width="{STROKE}" height="{STROKE}" '
            f'fill="{BLACK}"/>'
        )
    return "\n".join(parts)


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
            rotate(corner(WALL_COLOUR, inner_square=True), deg)
        )

    for n, colour in SNAKE_COLOURS.items():
        for facing, deg in DIRECTIONS.items():
            files[f"snake{n}_head_{facing}.svg"] = svg_wrapper(
                rotate(half_capsule(colour), deg)
            )
            files[f"snake{n}_tail_{facing}.svg"] = svg_wrapper(
                rotate(half_capsule(colour), deg)
            )
        for facing, deg in ("h", 0), ("v", 90):
            files[f"snake{n}_body_{facing}.svg"] = svg_wrapper(
                rotate(pipe(colour), deg)
            )
        for joining, deg in CORNER_JOININGS.items():
            files[f"snake{n}_corner_{joining}.svg"] = svg_wrapper(
                rotate(corner(colour, inner_square=True), deg)
            )

    for name, content in files.items():
        (OUT_DIR / name).write_text(content)
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
