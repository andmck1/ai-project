import typer
from src.cli.game import app as game_app
from src.cli.ai_training import app as ai_app

app = typer.Typer(help="Top level CLI")
app.add_typer(game_app, help="Snake CLI", name="game")
app.add_typer(ai_app, help="MLflow Experiment CLI", name="ai")
