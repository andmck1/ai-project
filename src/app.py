from src.game import Screen, Game


def run(
    is_draw_enabled: bool = True,
    framerate: int = 1,
    is_ai_random: bool = True,
    is_ai_enabled: bool = True,
) -> None:
    screen = Screen(is_draw_enabled=is_draw_enabled)
    game = Game(
        screen,
        "MAT501 - Coursework",
        is_draw_enabled=is_draw_enabled,
        framerate=framerate,
        is_ai_random=is_ai_random,
        is_ai_enabled=is_ai_enabled,
    )
    while game.frame():
        pass
