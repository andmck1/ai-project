from src.game.screen import Screen
from src.game.game import Game

import typer
from pathlib import Path

app = typer.Typer()
__run_app = typer.Typer()
app.add_typer(__run_app, name="run")


@__run_app.command()
def custom(
    is_draw_enabled: bool = True,
    framerate: int = 3,
    ai_model_path: str = "./models/xgboost.ubj",
    is_ai_random: bool = True,
    is_ai_enabled: bool = True,
    is_debug_enabled: bool = False,
) -> None:
    screen = Screen(is_draw_enabled=is_draw_enabled)
    ai_model_path_safe = Path(ai_model_path)
    game = Game(
        screen,
        "MAT501 - Coursework",
        is_draw_enabled=is_draw_enabled,
        framerate=framerate,
        ai_model_path=ai_model_path_safe,
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
def normal(ai_model_path: str = "./models/xgboost.ubj") -> None:
    custom(
        ai_model_path=ai_model_path,
        is_draw_enabled=True,
        is_ai_random=False,
        is_ai_enabled=True,
        is_debug_enabled=True,
    )
