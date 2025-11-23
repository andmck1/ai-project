import typer
from src.cli.game import app as game_app

app = typer.Typer(help="Top level CLI")
app.add_typer(game_app, help="Snake CLI", name="game")
