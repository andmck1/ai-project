from src.game import Screen, Game


def run(is_draw_enabled: bool = True, tick_length: int = 8) -> None:
    screen = Screen(is_draw_enabled=is_draw_enabled)
    game = Game(
        screen,
        "MAT501 - Coursework",
        is_draw_enabled=is_draw_enabled,
        tick_length=tick_length,
    )
    while game.frame():
        pass
