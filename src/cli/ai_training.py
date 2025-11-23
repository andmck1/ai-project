from src.ai.train_xgboost import run_experiment, ExperimentContext

import typer

app = typer.Typer()


@app.command(name="run-experiment")
def run_mlflow_experiment(
    number_episodes_random_simulation: int = 20,
    max_steps_random_simulation: int = 50,
    number_episodes_model_simulation: int = 20,
    max_steps_model_simulation: int = 1000,
) -> None:
    experiment_context = ExperimentContext(
        number_episodes_random_simulation=number_episodes_random_simulation,
        max_steps_random_simulation=max_steps_random_simulation,
        number_episodes_model_simulation=number_episodes_model_simulation,
        max_steps_model_simulation=max_steps_model_simulation,
    )
    run_experiment(experiment_context)
