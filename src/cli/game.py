import typer
from src.game import Screen, Game

app = typer.Typer()
__run_app = typer.Typer()
app.add_typer(__run_app, name="run")


@__run_app.command()
def custom(
    is_draw_enabled: bool = True,
    framerate: int = 3,
    is_ai_random: bool = True,
    is_ai_enabled: bool = True,
    is_debug_enabled: bool = False,
) -> None:
    screen = Screen(is_draw_enabled=is_draw_enabled)
    game = Game(
        screen,
        "MAT501 - Coursework",
        is_draw_enabled=is_draw_enabled,
        framerate=framerate,
        is_ai_random=is_ai_random,
        is_ai_enabled=is_ai_enabled,
        is_debug_enabled=is_debug_enabled,
    )
    while game.frame():
        pass


@__run_app.command()
def no_ai() -> None:
    custom(
        is_draw_enabled=True,
        is_ai_random=False,
        is_ai_enabled=False,
        is_debug_enabled=True,
    )


@__run_app.command()
def normal() -> None:
    custom(
        is_draw_enabled=True,
        is_ai_random=False,
        is_ai_enabled=True,
        is_debug_enabled=True,
    )
