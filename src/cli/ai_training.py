from src.config.logging import create_logger
from src.ai.train import run_experiment, ExperimentContext

import typer
from pathlib import Path
import yaml

logger = create_logger(__name__)

app = typer.Typer()


@app.command(name="run-experiment")
def run_mlflow_experiment(config_path: Path) -> None:
    logger.info("Reading from file: " + str(config_path.absolute()))
    with config_path.open("r") as config_file:
        config = yaml.safe_load(config_file)
        experiment_context = ExperimentContext(
            experiment_name=config["experiment_name"],
            train_validation_split=config["train_validation_split"],
            n_cross_validation_splits=config["n_cross_validation_splits"],
            number_episodes_random_simulation=config[
                "number_episodes_random_simulation"
            ],
            max_steps_random_simulation=config["max_steps_random_simulation"],
            max_number_model_runs=config["max_number_model_runs"],
            max_test_simulation_episodes=config[
                "max_test_simulation_episodes"
            ],
            max_n_test_simulation_steps=config["max_n_test_simulation_steps"],
            model_search=config["model_parameter_searches"],
        )

    run_experiment(experiment_context)
