from src.game.screen import Screen
from src.game.game import Game

import typer

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
    ai_mlflow_model_path_or_uri: str = "",
    is_two_ai_playing: bool = False,
) -> None:
    screen = Screen(is_draw_enabled=is_draw_enabled)
    game = Game(
        screen,
        "MAT501 - Coursework",
        is_draw_enabled=is_draw_enabled,
        framerate=framerate,
        ai_mlflow_model_path_or_uri=ai_mlflow_model_path_or_uri,
        is_ai_random=is_ai_random,
        is_ai_enabled=is_ai_enabled,
        is_debug_enabled=is_debug_enabled,
        is_two_ai_playing=is_two_ai_playing,
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
def only_ai(ai_model_mlflow_run_id: str) -> None:
    custom(
        ai_mlflow_model_path_or_uri=ai_model_mlflow_run_id,
        framerate=15,
        is_draw_enabled=True,
        is_ai_random=False,
        is_ai_enabled=True,
        is_debug_enabled=True,
        is_two_ai_playing=True,
    )


@__run_app.command()
def normal(ai_model_mlflow_run_id: str) -> None:
    custom(
        ai_mlflow_model_path_or_uri=ai_model_mlflow_run_id,
        is_draw_enabled=True,
        is_ai_random=False,
        is_ai_enabled=True,
        is_debug_enabled=True,
        framerate=10,
    )
